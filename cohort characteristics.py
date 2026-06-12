import os
import pandas as pd
import numpy as np
from scipy.stats import mannwhitneyu, chi2_contingency, fisher_exact
import warnings
warnings.filterwarnings('ignore')

base_dir = '/Users/cagriozdemir/PycharmProjects/Machine learning-based relapse prediction '

# ══════════════════════════════════════════════════════
# Load
# ══════════════════════════════════════════════════════
def load_core(dataset):
    path = lambda f: os.path.join(base_dir, 'datasets', dataset, f)
    return {
        'labels' : pd.read_csv(path('response.csv'),           index_col=0),
        'snv'    : pd.read_csv(path('snv_data.csv'),           index_col=0),
        'cadd'   : pd.read_csv(path('cadd_features.csv'),      index_col=0),
        'clin'   : pd.read_csv(path('clinical_features.csv'),  index_col=0),
        'cna'    : pd.read_csv(path('cna_features.csv'),       index_col=0),
    }

msk  = load_core('MSK_data')
meta = load_core('METABRIC_data')

CORE_GENES = ['TP53', 'PIK3CA']

# ══════════════════════════════════════════════════════
# Build full dataframe per cohort
# ══════════════════════════════════════════════════════
def build_df(data, genes):
    labels = data['labels']
    snv    = data['snv']
    cadd   = data['cadd']
    clin   = data['clin']
    cna    = data['cna']
    idx    = labels.index

    df            = pd.DataFrame(index=idx)
    df['relapse'] = labels['relapse']

    for col in clin.columns:
        df[col] = clin.reindex(idx)[col]

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
# Statistical test helpers
# ══════════════════════════════════════════════════════
def format_p(p):
    if pd.isna(p):      return 'N/A'
    if p < 0.001:       return f'{p:.2e} ***'
    elif p < 0.01:      return f'{p:.4f}  **'
    elif p < 0.05:      return f'{p:.4f}   *'
    else:               return f'{p:.4f}  ns'

def mwu_test(a, b):
    a, b = a.dropna(), b.dropna()
    if len(a) < 3 or len(b) < 3: return np.nan
    _, p = mannwhitneyu(a, b, alternative='two-sided')
    return p

def chi2_or_fisher(a, b):
    a, b = a.dropna(), b.dropna()
    combined = pd.concat([
        pd.Series(a.values).to_frame('val').assign(grp='R'),
        pd.Series(b.values).to_frame('val').assign(grp='NR')
    ])
    ct = pd.crosstab(combined['val'], combined['grp'])
    if ct.shape[0] < 2: return np.nan
    _, p, _, _ = chi2_contingency(ct.values)
    return p

# ══════════════════════════════════════════════════════
# Format helpers
# ══════════════════════════════════════════════════════
def fmt_mean(v):
    v = v.dropna()
    return f"{v.mean():.2f} ({v.std():.2f})"

def fmt_n(v, total):
    n = int(v)
    return f"{n} ({n/total*100:.1f}%)"

# ══════════════════════════════════════════════════════
# Print cohort characteristics table
# ══════════════════════════════════════════════════════
def print_cohort_table(df, cohort_name):
    rel  = df[df['relapse'] == 1]
    nonr = df[df['relapse'] == 0]
    n    = len(df)
    n_r  = len(rel)
    n_nr = len(nonr)

    W = 110
    print(f"\n{'='*W}")
    print(f"── Cohort Characteristics: {cohort_name}")
    print(f"{'='*W}")
    print(f"  {'Feature':<35} {'Total (n='+str(n)+')':<24} {'Relapse (n='+str(n_r)+')':<24} {'Non-Relapse (n='+str(n_nr)+')':<24} p-value")
    print(f"  {'-'*(W-2)}")

    # ── n total ───────────────────────────────────────
    print(f"  {'n':<35} {str(n):<24} {str(n_r):<24} {str(n_nr):<24}")
    print(f"  {'-'*(W-2)}")

    # ── Continuous features ───────────────────────────
    CONTINUOUS = ['Age', 'TMB', 'Fraction_Genome_Altered',
                  'TP53_cadd', 'PIK3CA_cadd',
                  'CCND1_cna_raw', 'MDM4_cna_raw',
                  'TP53_snv', 'PIK3CA_snv']

    for feat in CONTINUOUS:
        if feat not in df.columns: continue
        p = mwu_test(rel[feat], nonr[feat])
        print(f"  {feat+', mean (SD)':<35} {fmt_mean(df[feat]):<24} "
              f"{fmt_mean(rel[feat]):<24} {fmt_mean(nonr[feat]):<24} {format_p(p)}")

    print(f"  {'-'*(W-2)}")

    # ── ER Status ─────────────────────────────────────
    for feat, labels_map in [
        ('ER_Status',         {1: 'Positive', 0: 'Negative'}),
        ('PR_Status',         {1: 'Positive', 0: 'Negative'}),
        ('HER2_Status',       {1: 'Positive', 0: 'Negative'}),
        ('Menopausal_Status', {0: 'Pre',       1: 'Post'}),
    ]:
        if feat not in df.columns: continue
        p = chi2_or_fisher(rel[feat], nonr[feat])
        print(f"  {feat+', n (%)':<35} {'':<24} {'':<24} {'':<24} {format_p(p)}")
        for val, lbl in labels_map.items():
            t  = (df[feat]   == val).sum()
            r  = (rel[feat]  == val).sum()
            nr = (nonr[feat] == val).sum()
            print(f"    {lbl:<33} {fmt_n(t,n):<24} {fmt_n(r,n_r):<24} {fmt_n(nr,n_nr):<24}")

    print(f"  {'-'*(W-2)}")

    # ── Grade ─────────────────────────────────────────
    if 'Grade' in df.columns:
        p = chi2_or_fisher(rel['Grade'], nonr['Grade'])
        print(f"  {'Grade, n (%)':<35} {'':<24} {'':<24} {'':<24} {format_p(p)}")
        for val, lbl in [(1,'Grade 1'), (2,'Grade 2'), (3,'Grade 3')]:
            t  = (df['Grade'].round()   == val).sum()
            r  = (rel['Grade'].round()  == val).sum()
            nr = (nonr['Grade'].round() == val).sum()
            print(f"    {lbl:<33} {fmt_n(t,n):<24} {fmt_n(r,n_r):<24} {fmt_n(nr,n_nr):<24}")

    print(f"  {'-'*(W-2)}")

    # ── Tumor Stage ───────────────────────────────────
    for feat in ['T_Stage', 'Tumor_Stage', 'Tumor Stage']:
        if feat not in df.columns: continue
        p = chi2_or_fisher(rel[feat], nonr[feat])
        print(f"  {feat+', n (%)':<35} {'':<24} {'':<24} {'':<24} {format_p(p)}")
        for val in sorted(df[feat].dropna().unique()):
            t  = (df[feat]   == val).sum()
            r  = (rel[feat]  == val).sum()
            nr = (nonr[feat] == val).sum()
            print(f"    {str(val):<33} {fmt_n(t,n):<24} {fmt_n(r,n_r):<24} {fmt_n(nr,n_nr):<24}")

    print(f"{'='*W}\n")

# ══════════════════════════════════════════════════════
# Run
# ══════════════════════════════════════════════════════
print_cohort_table(msk_df,  'MSK-IMPACT')
print_cohort_table(meta_df, 'METABRIC')

# ══════════════════════════════════════════════════════
# Molecular subtypes: Triple-Positive, HER2-Enriched, Triple-Negative
# ══════════════════════════════════════════════════════
def print_subtypes(df, cohort_name):
    rel  = df[df['relapse'] == 1]
    nonr = df[df['relapse'] == 0]
    n    = len(df)
    n_r  = len(rel)
    n_nr = len(nonr)

    # Require all three receptor columns
    required = ['ER_Status', 'PR_Status', 'HER2_Status']
    if not all(c in df.columns for c in required):
        print(f"Receptor columns missing for {cohort_name}")
        return

    def mask(d, er, pr, her2):
        return (d['ER_Status'] == er) & (d['PR_Status'] == pr) & (d['HER2_Status'] == her2)

    def luminal_b(d):
        return (
            ((d['ER_Status'] == 1) & (d['PR_Status'] == 0) & (d['HER2_Status'] == 0)) |
            ((d['ER_Status'] == 1) & (d['PR_Status'] == 0) & (d['HER2_Status'] == 1)) |
            ((d['ER_Status'] == 1) & (d['PR_Status'] == 1) & (d['HER2_Status'] == 1))
        )

    lum_a  = mask(df, 1, 1, 0)
    lum_b  = luminal_b(df)
    tnbc   = mask(df, 0, 0, 0)
    other  = ~(lum_a | lum_b | tnbc)

    subtypes = {
        'Luminal A (ER+/PR+/HER2-)'              : lum_a,
        'Luminal B (ER+/PR-/HER2- or ER+/any/HER2+)': lum_b,
        'Triple-Negative (ER-/PR-/HER2-)'        : tnbc,
        'Other'                                   : other,
    }

    p = chi2_or_fisher(rel['ER_Status'].astype(str) + rel['PR_Status'].astype(str) + rel['HER2_Status'].astype(str),
                       nonr['ER_Status'].astype(str) + nonr['PR_Status'].astype(str) + nonr['HER2_Status'].astype(str))

    print(f"\n── Molecular Subtypes: {cohort_name} ──────────────────────────────────")
    print(f"  {'Subtype':<40} {'Total':<20} {'Relapse':<20} {'Non-Relapse':<20} p-value")
    print(f"  {'-'*100}")
    print(f"  {'n':<40} {str(n):<20} {str(n_r):<20} {str(n_nr):<20}")

    for name, m in subtypes.items():
        t  = m.sum()
        r  = (m & (df['relapse'] == 1)).sum()
        nr = (m & (df['relapse'] == 0)).sum()
        p_sub = chi2_or_fisher(
            rel['ER_Status'].astype(str) + rel['PR_Status'].astype(str) + rel['HER2_Status'].astype(str),
            nonr['ER_Status'].astype(str) + nonr['PR_Status'].astype(str) + nonr['HER2_Status'].astype(str)
        )
        t_str  = f"{t} ({t/n*100:.1f}%)"
        r_str  = f"{r} ({r/n_r*100:.1f}%)"
        nr_str = f"{nr} ({nr/n_nr*100:.1f}%)"
        print(f"  {name:<40} {t_str:<20} {r_str:<20} {nr_str:<20}")

    # Individual p-values per subtype
    print(f"\n  Individual subtype p-values (Fisher's exact, relapse vs non-relapse):")
    for name, m in subtypes.items():
        r_in   = (m & (df['relapse'] == 1)).sum()
        r_out  = n_r - r_in
        nr_in  = (m & (df['relapse'] == 0)).sum()
        nr_out = n_nr - nr_in
        ct     = np.array([[r_in, r_out], [nr_in, nr_out]])
        _, p_sub, _, _ = chi2_contingency(ct)
        t_str  = f"{m.sum()} ({m.sum()/n*100:.1f}%)"
        r_str  = f"{r_in} ({r_in/n_r*100:.1f}%)"
        nr_str = f"{nr_in} ({nr_in/n_nr*100:.1f}%)"
        print(f"    {name:<45} total={t_str:<18} relapse={r_str:<18} non-relapse={nr_str:<18} {format_p(p_sub)}")

    print()

print_subtypes(msk_df,  'MSK-IMPACT')
print_subtypes(meta_df, 'METABRIC')
