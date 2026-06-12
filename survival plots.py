import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import logrank_test
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

base_dir   = '/Users/cagriozdemir/PycharmProjects/Machine learning-based relapse prediction '
CORE_GENES = ['TP53', 'PIK3CA']
DROP_CLIN  = ['T_Stage', 'N_Stage', 'Menopausal_Status', 'Age']

# ══════════════════════════════════════════════════════
# Load features
# ══════════════════════════════════════════════════════
def load_dataset(path, genes):
    snv  = pd.read_csv(os.path.join(path, 'snv_data.csv'),          index_col=0)
    cadd = pd.read_csv(os.path.join(path, 'cadd_features.csv'),     index_col=0)
    clin = pd.read_csv(os.path.join(path, 'clinical_features.csv'), index_col=0)
    lab  = pd.read_csv(os.path.join(path, 'response.csv'),          index_col=0)
    cna  = pd.read_csv(os.path.join(path, 'cna_features.csv'),      index_col=0)

    clin      = clin.drop(columns=DROP_CLIN, errors='ignore')
    avail_snv = [g for g in genes if g in snv.columns]
    cadd_cols = [c for c in cadd.columns]
    raw_cols  = [c for c in cna.columns if c.endswith('_cna_raw')]

    common = (snv.index
              .intersection(clin.index)
              .intersection(lab.index)
              .intersection(cadd.index)
              .intersection(cna.index))

    X = pd.concat([
        snv.loc[common, avail_snv].add_suffix('_snv'),
        cadd.reindex(common).fillna(0)[cadd_cols],
        clin.loc[common],
        cna.reindex(common).fillna(0)[raw_cols],
    ], axis=1).fillna(0)
    X = X.sort_index(axis=1)
    y = lab.loc[common, 'relapse'].values
    return X, y

MSK_DIR  = os.path.join(base_dir, 'datasets', 'MSK_data')
META_DIR = os.path.join(base_dir, 'datasets', 'METABRIC_data')

msk_X,  msk_y  = load_dataset(MSK_DIR,  CORE_GENES)
meta_X, meta_y = load_dataset(META_DIR, CORE_GENES)

print(f"MSK      : {msk_X.shape}  | relapse={msk_y.sum()}  non-relapse={(msk_y==0).sum()}")
print(f"METABRIC : {meta_X.shape}  | relapse={meta_y.sum()}  non-relapse={(meta_y==0).sum()}")

combined_X = pd.concat([msk_X, meta_X])
combined_y = np.concatenate([msk_y, meta_y])
print(f"Combined : {combined_X.shape}  | relapse={combined_y.sum()}")

# ══════════════════════════════════════════════════════
# Fixed RF hyperparameters (Exp C final model)
# ══════════════════════════════════════════════════════
RF_PARAMS = dict(
    n_estimators    = 300,
    max_depth       = 4,
    min_samples_leaf= 7,
    class_weight    = 'balanced',
    criterion       = 'entropy',
    random_state    = 42,
    n_jobs          = -1
)

# ══════════════════════════════════════════════════════
# 5-fold CV on combined cohort — out-of-fold predictions
# ══════════════════════════════════════════════════════
from sklearn.model_selection import StratifiedKFold

cv    = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
proba = np.zeros(len(combined_y))

print(f"\nRunning 5-fold CV on combined cohort (MSK+METABRIC)...")
for fold, (tr_idx, te_idx) in enumerate(cv.split(combined_X.values, combined_y)):
    rf = RandomForestClassifier(**RF_PARAMS)
    rf.fit(combined_X.values[tr_idx], combined_y[tr_idx])
    proba[te_idx] = rf.predict_proba(combined_X.values[te_idx])[:, 1]
    print(f"  Fold {fold+1}/5 done")

print(f"  Zeros in proba (should be 0): {(proba == 0).sum()}")

# ── Extract METABRIC patients only ────────────────────
n_msk        = len(msk_X)
meta_proba   = proba[n_msk:]

print(f"\nMETABRIC out-of-fold predictions:")
print(f"  High risk (>=0.5) : {(meta_proba >= 0.5).sum()}")
print(f"  Low risk  (<0.5)  : {(meta_proba < 0.5).sum()}")
print(f"  Mean probability  : {meta_proba.mean():.3f}")

# ══════════════════════════════════════════════════════
# Load METABRIC survival data
# ══════════════════════════════════════════════════════
meta_surv = pd.read_csv(
    os.path.join(base_dir, 'datasets', 'METABRIC_data', 'survival_labels.csv'),
    index_col=0
)

surv_df = meta_surv.loc[meta_X.index].copy()
surv_df['risk_prob']  = meta_proba
surv_df['risk_group'] = (meta_proba >= 0.5).astype(int)
surv_df['risk_label'] = surv_df['risk_group'].map({1: 'High Risk', 0: 'Low Risk'})

# ── Convert event columns to numeric ──────────────────
surv_df['Relapse Free Status'] = surv_df['Relapse Free Status'].map(
    {'0:Not Recurred': 0, '1:Recurred': 1}
).astype(float)
surv_df['Overall Survival Status'] = surv_df['Overall Survival Status'].map(
    {'0:LIVING': 0, '1:DECEASED': 1}
).astype(float)
surv_df['Relapse Free Status (Months)'] = pd.to_numeric(
    surv_df['Relapse Free Status (Months)'], errors='coerce'
)
surv_df['Overall Survival (Months)'] = pd.to_numeric(
    surv_df['Overall Survival (Months)'], errors='coerce'
)

DFS_TIME  = 'Relapse Free Status (Months)'
DFS_EVENT = 'Relapse Free Status'
OS_TIME   = 'Overall Survival (Months)'
OS_EVENT  = 'Overall Survival Status'

# ── Drop NaNs separately for DFS and OS ───────────────
surv_dfs = surv_df.dropna(subset=[DFS_TIME, DFS_EVENT]).copy()
surv_os  = surv_df.dropna(subset=[OS_TIME,  OS_EVENT]).copy()

print(f"\nDFS cohort : {len(surv_dfs)} patients | "
      f"high={(surv_dfs['risk_group']==1).sum()}  "
      f"low={(surv_dfs['risk_group']==0).sum()}")
print(f"OS  cohort : {len(surv_os)} patients | "
      f"high={(surv_os['risk_group']==1).sum()}  "
      f"low={(surv_os['risk_group']==0).sum()}")

# ── Save directory ─────────────────────────────────────
SAVE_DIR = os.path.join(base_dir, 'datasets', 'survival_plots')
os.makedirs(SAVE_DIR, exist_ok=True)

# ══════════════════════════════════════════════════════
# Kaplan-Meier helper
# ══════════════════════════════════════════════════════
def plot_km(time_col, event_col, title, save_path, df):
    high = df[df['risk_group'] == 1]
    low  = df[df['risk_group'] == 0]

    results = logrank_test(
        high[time_col], low[time_col],
        event_observed_A=high[event_col],
        event_observed_B=low[event_col]
    )
    p_value = results.p_value

    kmf_high = KaplanMeierFitter()
    kmf_low  = KaplanMeierFitter()

    fig, ax = plt.subplots(figsize=(10, 7))

    kmf_high.fit(high[time_col], high[event_col],
                 label=f'High Risk (n={len(high)})')
    kmf_high.plot_survival_function(ax=ax, color='#E74C3C',
                                    linewidth=2.5, ci_show=True, ci_alpha=0.1)

    kmf_low.fit(low[time_col], low[event_col],
                label=f'Low Risk (n={len(low)})')
    kmf_low.plot_survival_function(ax=ax, color='#2E86C1',
                                   linewidth=2.5, ci_show=True, ci_alpha=0.1)

    p_text = f'p = {p_value:.4f}' if p_value >= 0.0001 else 'p < 0.0001'
    ax.text(0.65, 0.79, p_text, transform=ax.transAxes,
            fontsize=17, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='white',
                      edgecolor='gray', alpha=0.8))

    ax.set_xlabel('Time (Months)', fontsize=22)
    ax.set_ylabel('Survival Probability', fontsize=22)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=18, loc='upper right')
    ax.grid(axis='y', linestyle=':', alpha=0.4)
    ax.tick_params(axis='both', labelsize=22)

    plt.tight_layout()
    plt.savefig(save_path, dpi=450, bbox_inches='tight')
    plt.close()

    print(f"\n── {title} ──────────────────────────────")
    print(f"  Log-rank p-value : {p_value:.6f}  "
          f"{'✓ YES' if p_value < 0.05 else '✗ NO'}")
    print(f"  High risk median : {kmf_high.median_survival_time_:.1f} months")
    print(f"  Low risk  median : {kmf_low.median_survival_time_:.1f} months")

    return p_value


# ══════════════════════════════════════════════════════
# Cox PH helper
# ══════════════════════════════════════════════════════
def run_cox(time_col, event_col, label, df):
    cox_df = pd.DataFrame({
        'duration'  : df[time_col],
        'event'     : df[event_col],
        'risk_group': df['risk_group']
    })
    cph = CoxPHFitter()
    cph.fit(cox_df, duration_col='duration', event_col='event')

    hr      = np.exp(cph.params_['risk_group'])
    ci_low  = np.exp(cph.confidence_intervals_['95% lower-bound']['risk_group'])
    ci_high = np.exp(cph.confidence_intervals_['95% upper-bound']['risk_group'])
    p       = cph.summary['p']['risk_group']

    print(f"\n── Cox PH — {label} ────────────────────────────")
    print(f"  Hazard Ratio (HR) : {hr:.3f}")
    print(f"  95% CI            : [{ci_low:.3f} - {ci_high:.3f}]")
    print(f"  p-value           : {p:.6f}  {'✓ YES' if p < 0.05 else '✗ NO'}")

    return hr, ci_low, ci_high, p

# ══════════════════════════════════════════════════════
# Run KM and Cox
# ══════════════════════════════════════════════════════
p_dfs = plot_km(
    time_col  = DFS_TIME,
    event_col = DFS_EVENT,
    title     = 'Disease Free Survival — METABRIC\n(Combined MSK+METABRIC, 5-fold CV)',
    save_path = os.path.join(SAVE_DIR, 'km_dfs_metabric.png'),
    df        = surv_dfs
)
hr_dfs, ci_l_dfs, ci_h_dfs, p_cox_dfs = run_cox(
    DFS_TIME, DFS_EVENT, 'DFS — METABRIC', surv_dfs
)

p_os = plot_km(
    time_col  = OS_TIME,
    event_col = OS_EVENT,
    title     = 'Overall Survival — METABRIC\n(Combined MSK+METABRIC, 5-fold CV)',
    save_path = os.path.join(SAVE_DIR, 'km_os_metabric.png'),
    df        = surv_os
)
hr_os, ci_l_os, ci_h_os, p_cox_os = run_cox(
    OS_TIME, OS_EVENT, 'OS — METABRIC', surv_os
)

# ══════════════════════════════════════════════════════
# Summary
# ══════════════════════════════════════════════════════
print(f"\n{'='*65}")
print(f"── Survival Analysis Summary ───────────────────────────────")
print(f"{'='*65}")
print(f"  Model    : RF (5-fold CV on combined cohort)")
print(f"  Test     : METABRIC ({len(meta_X)} patients)")
print(f"\n  ── DFS ({len(surv_dfs)} patients) ─────────────────────────────")
print(f"  Log-rank p : {p_dfs:.6f}  {'✓' if p_dfs < 0.05 else '✗'}")
print(f"  Cox HR     : {hr_dfs:.3f} [{ci_l_dfs:.3f} - {ci_h_dfs:.3f}]  p={p_cox_dfs:.6f}")
print(f"\n  ── OS ({len(surv_os)} patients) ──────────────────────────────")
print(f"  Log-rank p : {p_os:.6f}  {'✓' if p_os < 0.05 else '✗'}")
print(f"  Cox HR     : {hr_os:.3f} [{ci_l_os:.3f} - {ci_h_os:.3f}]  p={p_cox_os:.6f}")
print(f"\n  Plots saved to: {SAVE_DIR}")
print(f"{'='*65}")
