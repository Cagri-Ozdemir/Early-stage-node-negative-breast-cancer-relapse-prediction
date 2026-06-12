import os
import pandas as pd
import numpy as np
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier
import shap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy.stats import wilcoxon
import warnings
warnings.filterwarnings('ignore')

base_dir = '/Users/cagriozdemir/PycharmProjects/Machine learning-based relapse prediction '
MSK_DIR  = os.path.join(base_dir, 'datasets', 'MSK_data')
META_DIR = os.path.join(base_dir, 'datasets', 'METABRIC_data')
FIG_DIR  = os.path.join(base_dir, 'datasets', 'shap_plots')
os.makedirs(FIG_DIR, exist_ok=True)

CORE_GENES = ['TP53', 'PIK3CA']
DROP_CLIN  = ['T_Stage', 'N_Stage']

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
    common = (snv.index.intersection(clin.index).intersection(lab.index)
              .intersection(cadd.index).intersection(cna.index))
    X = pd.concat([
        snv.loc[common, avail_snv].add_suffix('_snv'),
        cadd.reindex(common).fillna(0)[cadd_cols],
        clin.loc[common],
        cna.reindex(common).fillna(0)[raw_cols],
    ], axis=1).fillna(0)
    X = X.sort_index(axis=1)
    y = lab.loc[common, 'relapse'].values
    return X, y

msk_X,  msk_y  = load_dataset(MSK_DIR,  CORE_GENES)
meta_X, meta_y = load_dataset(META_DIR, CORE_GENES)
combined_X = pd.concat([msk_X, meta_X])
combined_y = np.concatenate([msk_y, meta_y])

print(f"Combined : {combined_X.shape}  | relapse={combined_y.sum()}")
print(f"Features : {combined_X.columns.tolist()}")

CLIN_FEATURES = [c for c in combined_X.columns if c in [
    'Age', 'Menopausal_Status', 'Grade', 'ER_Status', 'PR_Status', 'HER2_Status'
]]
GENOMIC_FEATURES = [c for c in combined_X.columns if c not in CLIN_FEATURES]

print(f"\nClinicopathological features ({len(CLIN_FEATURES)}): {CLIN_FEATURES}")
print(f"Genomic features ({len(GENOMIC_FEATURES)}): {GENOMIC_FEATURES}")

RENAME_MAP = {
    'Fraction_Genome_Altered': 'FGA',
    'Age': 'Age',
    'TMB': 'TMB',
    'Grade': 'Grade',
    'CCND1_cna_raw': 'CCND1 CNAs',
    'MDM4_cna_raw': 'MDM4 CNAs',
    'ER_Status': 'ER Status',
    'TP53_cadd': 'TP53 CADD',
    'PIK3CA_cadd': 'PIK3CA CADD',
    'PIK3CA_snv': 'PIK3CA SNVs',
    'PR_Status': 'PR Status',
    'HER2_Status': 'HER2 Status',
    'Menopausal_Status': 'Menopausal Status',
    'TP53_snv': 'TP53 SNVs',
}

# ══════════════════════════════════════════════════════
# RF hyperparameters (Exp C final model)
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

COLOR_CLIN    = '#2CA02C'
COLOR_GENOMIC = '#D62728'

# ══════════════════════════════════════════════════════
# PART 1: Train on ALL combined data → SHAP analysis
# ══════════════════════════════════════════════════════
print(f"\n── Training RF on full combined cohort for SHAP ──────────")

combined_X_renamed = combined_X.rename(columns=RENAME_MAP)
CLIN_FEATURES_R    = [RENAME_MAP.get(f, f) for f in CLIN_FEATURES]

rf_full     = RandomForestClassifier(**RF_PARAMS)
rf_full.fit(combined_X_renamed.values, combined_y)
explainer   = shap.TreeExplainer(rf_full)
shap_values = explainer.shap_values(combined_X_renamed.values)

# For RF, shap_values shape can be (n, features, 2) or list [class0, class1]
if isinstance(shap_values, list):
    shap_vals = shap_values[1]
elif shap_values.ndim == 3:
    shap_vals = shap_values[:, :, 1]
else:
    shap_vals = shap_values

shap_df   = pd.DataFrame(np.abs(shap_vals), columns=combined_X_renamed.columns)
mean_shap = shap_df.mean().sort_values(ascending=False)

# ── Bar plot ───────────────────────────────────────────
fig, ax    = plt.subplots(figsize=(10, 8))
sorted_mean = mean_shap.sort_values(ascending=True)
colors      = [COLOR_CLIN if f in CLIN_FEATURES_R else COLOR_GENOMIC for f in sorted_mean.index]
ax.barh(range(len(sorted_mean)), sorted_mean.values, color=colors,
        edgecolor='white', linewidth=0.3)
ax.set_yticks(range(len(sorted_mean)))
ax.set_yticklabels(sorted_mean.index)
ax.set_xlabel('Mean |SHAP Value|', fontsize=22)
ax.tick_params(labelsize=22)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
legend_elements = [
    Patch(facecolor=COLOR_CLIN,    label='Clinicopathological'),
    Patch(facecolor=COLOR_GENOMIC, label='Genomic'),
]
ax.legend(handles=legend_elements, fontsize=20, frameon=False, loc='lower right')
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'shap_bar_all_features.png'), dpi=450, bbox_inches='tight')
plt.close()

# ── Pie chart ──────────────────────────────────────────
clin_shap    = mean_shap[CLIN_FEATURES_R].sum()
genomic_shap = mean_shap[[f for f in mean_shap.index if f not in CLIN_FEATURES_R]].sum()
total        = clin_shap + genomic_shap
clin_pct     = clin_shap / total * 100
genomic_pct  = genomic_shap / total * 100

fig, ax = plt.subplots(figsize=(7, 7))
wedges, texts, autotexts = ax.pie(
    [clin_pct, genomic_pct],
    labels=[f'Clinicopathological\n({len(CLIN_FEATURES_R)} features)',
            f'Genomic\n({len(GENOMIC_FEATURES)} features)'],
    colors=[COLOR_CLIN, COLOR_GENOMIC],
    autopct='%1.1f%%', startangle=90,
    textprops={'fontsize': 17}
)
for autotext in autotexts:
    autotext.set_fontsize(17)
    autotext.set_fontweight('bold')
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'shap_pie_clinical_genomic.png'), dpi=450, bbox_inches='tight')
plt.close()

# ── Beeswarm ───────────────────────────────────────────
shap_exp = shap.Explanation(
    values        = shap_vals,
    base_values   = explainer.expected_value[1] if isinstance(explainer.expected_value, np.ndarray) else explainer.expected_value,
    data          = combined_X_renamed.values,
    feature_names = combined_X_renamed.columns.tolist()
)

plt.figure(figsize=(12, 8))
shap.plots.beeswarm(shap_exp, max_display=14, show=False)

fig = plt.gcf()
for a in fig.get_axes():
    if a != plt.gca():
        a.tick_params(labelsize=16)
        a.yaxis.label.set_size(16)

plt.xlabel(plt.gca().get_xlabel(), fontsize=22)
plt.ylabel(plt.gca().get_ylabel(), fontsize=22)
plt.tick_params(axis='both', labelsize=19)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'shap_beeswarm.png'), dpi=450, bbox_inches='tight')
plt.close()

print(f"SHAP plots saved to: {FIG_DIR}")

# ══════════════════════════════════════════════════════
# PART 2: Ablation — 5-fold CV x5
# ══════════════════════════════════════════════════════
print(f"\n── Ablation Analysis ──────────────────────────────────────")

def run_ablation_cv(X, y, label, n_splits=5, n_repeats=5):
    cv   = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=42)
    aucs = []
    for tr_idx, te_idx in cv.split(X, y):
        m = RandomForestClassifier(**RF_PARAMS)
        m.fit(X[tr_idx], y[tr_idx])
        proba = m.predict_proba(X[te_idx])[:, 1]
        aucs.append(roc_auc_score(y[te_idx], proba))
    aucs = np.array(aucs)
    print(f"  {label:<30} AUC: {aucs.mean():.3f} +- {aucs.std():.3f}")
    return aucs

X_clin    = combined_X[CLIN_FEATURES].values
X_genomic = combined_X[GENOMIC_FEATURES].values
X_all     = combined_X.values

aucs_clin    = run_ablation_cv(X_clin,    combined_y, 'Clinicopathological only')
aucs_genomic = run_ablation_cv(X_genomic, combined_y, 'Genomic only')
aucs_all     = run_ablation_cv(X_all,     combined_y, 'All features')

# ── Box plot ───────────────────────────────────────────
COLOR_CLIN2    = '#2CA02C'
COLOR_GENOMIC2 = '#D62728'
COLOR_ALL      = '#1F77B4'

fig, ax = plt.subplots(figsize=(11, 8))
data   = [aucs_clin, aucs_genomic, aucs_all]
labels = ['Clinicopathological\nonly', 'Genomic\nonly', 'All\nfeatures']
colors = [COLOR_CLIN2, COLOR_GENOMIC2, COLOR_ALL]

positions = [1, 1.7, 2.4]
bp = ax.boxplot(data, positions=positions, patch_artist=True, showfliers=False, widths=0.38,
                medianprops=dict(color='white', linewidth=2.9),
                whiskerprops=dict(linewidth=1.6),
                capprops=dict(linewidth=1.6))

for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
for i, color in enumerate(colors):
    bp['whiskers'][i*2].set_color(color)
    bp['whiskers'][i*2+1].set_color(color)
    bp['caps'][i*2].set_color(color)
    bp['caps'][i*2+1].set_color(color)

jitter = 0.07
for pos, (aucs, color) in zip(positions, zip(data, colors)):
    ax.scatter(
        np.full(len(aucs), pos) + np.random.uniform(-jitter, jitter, len(aucs)),
        aucs, color=color, s=80, alpha=0.6, zorder=3,
        edgecolors='white', linewidths=0.7
    )

def pval_label(p):
    if p < 0.001: return '***'
    elif p < 0.01: return '**'
    elif p < 0.05: return '*'
    else: return 'ns'

y_max = max(aucs_all.max(), aucs_genomic.max(), aucs_clin.max())

_, p_genomic = wilcoxon(aucs_all, aucs_genomic)
y1 = y_max + 0.010
ax.plot([1.7, 1.7, 2.4, 2.4], [y1, y1+0.004, y1+0.004, y1], color='black', linewidth=1.3)
ax.text(2.05, y1+0.005, pval_label(p_genomic), ha='center', va='bottom', fontsize=22, fontweight='bold')

_, p_clin = wilcoxon(aucs_all, aucs_clin)
y2 = y1 + 0.028
ax.plot([1, 1, 2.4, 2.4], [y2, y2+0.004, y2+0.004, y2], color='black', linewidth=1.3)
ax.text(1.7, y2+0.005, pval_label(p_clin), ha='center', va='bottom', fontsize=22, fontweight='bold')

ax.set_xticks(positions)
ax.set_xticklabels(labels, fontsize=25)
ax.set_ylabel('AUC', fontsize=28, labelpad=10)
ax.tick_params(axis='y', labelsize=28)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_ylim(0.60, y2+0.06)

plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'ablation_auc_boxplot.png'), dpi=450, bbox_inches='tight')
plt.close()

print(f"\nAblation box plot saved to: {FIG_DIR}")
print(f"\n-- Done --------------------------------------------------")
