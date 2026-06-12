import pandas as pd
import numpy as np
from scipy.stats import mannwhitneyu, chi2_contingency, fisher_exact
import os

base_dir   = '/Users/cagriozdemir/PycharmProjects/Machine learning-based relapse prediction '
CORE_GENES = ['TP53', 'PIK3CA']

# ══════════════════════════════════════════════════════
# Load from saved files
# ══════════════════════════════════════════════════════
def load_core(dataset):
    path = lambda f: os.path.join(base_dir, 'datasets', dataset, f)
    return {
        'surv'   : pd.read_csv(path('survival_labels.csv'),   index_col=0),
        'labels' : pd.read_csv(path('response.csv'),           index_col=0),
        'snv'    : pd.read_csv(path('snv_data.csv'),           index_col=0),
        'cadd'   : pd.read_csv(path('cadd_features.csv'),      index_col=0),
        'clin'   : pd.read_csv(path('clinical_features.csv'),  index_col=0),
        'cna'    : pd.read_csv(path('cna_features.csv'),       index_col=0),
    }

msk  = load_core('MSK_data')
meta = load_core('METABRIC_data')

# ══════════════════════════════════════════════════════
# Build display dataframe
# ══════════════════════════════════════════════════════
def build_df(data, genes):
    labels = data['labels']
    surv   = data['surv']
    snv    = data['snv']
    cadd   = data['cadd']
    clin   = data['clin']
    cna    = data['cna']
    idx    = labels.index

    df            = pd.DataFrame(index=idx)
    df['relapse'] = labels['relapse']

    for col in clin.columns:
        df[col] = clin.reindex(idx)[col]
    for col in surv.columns:
        df[col] = surv.reindex(idx)[col]
    for gene in genes:
        if gene in snv.columns:
            df[f'{gene}_snv'] = snv.reindex(idx)[gene].fillna(0)
    for gene in genes:
        col = f'{gene}_cadd'
        if col in cadd.columns:
            df[f'{gene}_cadd'] = cadd.reindex(idx)[col].fillna(0)
    for col in cna.columns:
        df[col] = cna.reindex(idx)[col].fillna(0)

    return df

msk_df  = build_df(msk,  CORE_GENES)
meta_df = build_df(meta, CORE_GENES)

# ══════════════════════════════════════════════════════
# Define binary vs continuous features
# ══════════════════════════════════════════════════════
# Binary features: use chi-square (or Fisher's exact if small counts)
BINARY_FEATURES = [
    'ER_Status', 'PR_Status', 'HER2_Status',
    'Menopausal_Status',
    'TP53_snv', 'PIK3CA_snv'
]

# Ordinal / continuous features: use Mann-Whitney U
CONTINUOUS_FEATURES = [
    'Age', 'TMB', 'Fraction_Genome_Altered',
    'Grade',
    'TP53_cadd', 'PIK3CA_cadd',
    'CCND1_cna_raw', 'MDM4_cna_raw'
]

# ══════════════════════════════════════════════════════
# Statistical test functions
# ══════════════════════════════════════════════════════
def format_p(p):
    if p < 0.001:   return f'{p:.2e} ***'
    elif p < 0.01:  return f'{p:.4f}  **'
    elif p < 0.05:  return f'{p:.4f}   *'
    else:           return f'{p:.4f}  ns'

def mw_test(a, b):
    a = a.dropna()
    b = b.dropna()
    if len(a) < 3 or len(b) < 3:
        return np.nan
    _, p = mannwhitneyu(a, b, alternative='two-sided')
    return p

def chi2_test(a, b):
    a = a.dropna()
    b = b.dropna()
    combined = pd.concat([
        pd.Series(a.values, name='val').to_frame().assign(cohort='A'),
        pd.Series(b.values, name='val').to_frame().assign(cohort='B')
    ])
    ct = pd.crosstab(combined['val'], combined['cohort'])
    if ct.shape[0] < 2:
        return np.nan
    # Use Fisher's exact for 2x2, chi2 otherwise
    if ct.shape == (2, 2):
        _, p = fisher_exact(ct.values)
    else:
        _, p, _, _ = chi2_contingency(ct.values)
    return p

# ══════════════════════════════════════════════════════
# Run comparison: relapse vs relapse, non-relapse vs non-relapse
# ══════════════════════════════════════════════════════
msk_rel   = msk_df[msk_df['relapse'] == 1]
msk_nonr  = msk_df[msk_df['relapse'] == 0]
meta_rel  = meta_df[meta_df['relapse'] == 1]
meta_nonr = meta_df[meta_df['relapse'] == 0]

print(f"\n{'='*85}")
print(f"Distribution Comparison: MSK vs METABRIC")
print(f"{'='*85}")
print(f"{'Feature':<20} {'Test':<12} {'Relapse p-value':<25} {'Non-Relapse p-value'}")
print(f"{'-'*85}")

results = []

for feat in BINARY_FEATURES:
    msk_r_col  = msk_rel[feat]   if feat in msk_df.columns  else None
    msk_nr_col = msk_nonr[feat]  if feat in msk_df.columns  else None
    meta_r_col = meta_rel[feat]  if feat in meta_df.columns else None
    meta_nr_col= meta_nonr[feat] if feat in meta_df.columns else None

    if msk_r_col is None or meta_r_col is None:
        print(f"  {feat:<18} {'Chi2':<12} {'N/A':<25} {'N/A'}")
        continue

    p_rel  = chi2_test(msk_r_col,  meta_r_col)
    p_nonr = chi2_test(msk_nr_col, meta_nr_col)
    print(f"  {feat:<18} {'Chi2/Fisher':<12} {format_p(p_rel):<25} {format_p(p_nonr)}")
    results.append({'Feature': feat, 'Test': 'Chi2/Fisher',
                    'Relapse_p': p_rel, 'NonRelapse_p': p_nonr})

for feat in CONTINUOUS_FEATURES:
    msk_r_col  = msk_rel[feat]   if feat in msk_df.columns  else None
    msk_nr_col = msk_nonr[feat]  if feat in msk_df.columns  else None
    meta_r_col = meta_rel[feat]  if feat in meta_df.columns else None
    meta_nr_col= meta_nonr[feat] if feat in meta_df.columns else None

    if msk_r_col is None or meta_r_col is None:
        print(f"  {feat:<18} {'MWU':<12} {'N/A':<25} {'N/A'}")
        continue

    p_rel  = mw_test(msk_r_col,  meta_r_col)
    p_nonr = mw_test(msk_nr_col, meta_nr_col)
    print(f"  {feat:<18} {'MWU':<12} {format_p(p_rel):<25} {format_p(p_nonr)}")
    results.append({'Feature': feat, 'Test': 'MWU',
                    'Relapse_p': p_rel, 'NonRelapse_p': p_nonr})

print(f"{'='*85}")

# # Save results
# results_df = pd.DataFrame(results)
# out_path = os.path.join(base_dir, 'datasets', 'distribution_comparison.csv')
# results_df.to_csv(out_path, index=False)
# print(f"\nSaved → {out_path}")
# ══════════════════════════════════════════════════════
# Within-cohort: relapse vs non-relapse comparison
# ══════════════════════════════════════════════════════
print(f"\n{'='*85}")
print(f"Within-Cohort: Relapse vs Non-Relapse")
print(f"{'='*85}")
print(f"{'Feature':<25} {'Test':<12} {'MSK p-value':<25} {'METABRIC p-value'}")
print(f"{'-'*85}")

for feat in BINARY_FEATURES:
    if feat not in msk_df.columns or feat not in meta_df.columns:
        print(f"  {feat:<23} {'Chi2/Fisher':<12} {'N/A':<25} {'N/A'}")
        continue
    p_msk  = chi2_test(msk_df[feat],  msk_df['relapse'])  # feature vs relapse label
    p_meta = chi2_test(meta_df[feat], meta_df['relapse'])
    print(f"  {feat:<23} {'Chi2/Fisher':<12} {format_p(p_msk):<25} {format_p(p_meta)}")

for feat in CONTINUOUS_FEATURES:
    if feat not in msk_df.columns or feat not in meta_df.columns:
        print(f"  {feat:<23} {'MWU':<12} {'N/A':<25} {'N/A'}")
        continue
    msk_r   = msk_df[msk_df['relapse']==1][feat]
    msk_nr  = msk_df[msk_df['relapse']==0][feat]
    meta_r  = meta_df[meta_df['relapse']==1][feat]
    meta_nr = meta_df[meta_df['relapse']==0][feat]
    p_msk  = mw_test(msk_r,  msk_nr)
    p_meta = mw_test(meta_r, meta_nr)
    print(f"  {feat:<23} {'MWU':<12} {format_p(p_msk):<25} {format_p(p_meta)}")

print(f"{'='*85}")
