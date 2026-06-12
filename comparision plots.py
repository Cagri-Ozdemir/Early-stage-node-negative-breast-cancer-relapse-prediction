import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold
from scipy.stats import wilcoxon
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from xgboost import XGBClassifier
import sklearn.base
import warnings
from sklearn.tree import DecisionTreeClassifier
warnings.filterwarnings('ignore')

base_dir = '/Users/cagriozdemir/PycharmProjects/Machine learning-based relapse prediction '
MSK_DIR  = os.path.join(base_dir, 'datasets', 'MSK_data')
META_DIR = os.path.join(base_dir, 'datasets', 'METABRIC_data')

CORE_GENES = ['TP53', 'PIK3CA']

# ══════════════════════════════════════════════════════
# Load with configurable DROP_CLIN
# ══════════════════════════════════════════════════════
def load_dataset(path, genes, drop_clin):
    snv  = pd.read_csv(os.path.join(path, 'snv_data.csv'),          index_col=0)
    cadd = pd.read_csv(os.path.join(path, 'cadd_features.csv'),     index_col=0)
    clin = pd.read_csv(os.path.join(path, 'clinical_features.csv'), index_col=0)
    lab  = pd.read_csv(os.path.join(path, 'response.csv'),          index_col=0)
    cna  = pd.read_csv(os.path.join(path, 'cna_features.csv'),      index_col=0)

    clin      = clin.drop(columns=drop_clin, errors='ignore')
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

# ── Condition 1: All features ──────────────────────────
DROP_ALL  = ['T_Stage', 'N_Stage']
msk_X_all,  msk_y_all  = load_dataset(MSK_DIR,  CORE_GENES, DROP_ALL)
meta_X_all, meta_y_all = load_dataset(META_DIR, CORE_GENES, DROP_ALL)
combined_X_all = pd.concat([msk_X_all, meta_X_all])
combined_y_all = np.concatenate([msk_y_all, meta_y_all])

# ── Condition 2: Exclude Age + Menopausal_Status ───────
DROP_EXCL = ['T_Stage', 'N_Stage', 'Menopausal_Status', 'Age']
msk_X_excl,  msk_y_excl  = load_dataset(MSK_DIR,  CORE_GENES, DROP_EXCL)
meta_X_excl, meta_y_excl = load_dataset(META_DIR, CORE_GENES, DROP_EXCL)
combined_X_excl = pd.concat([msk_X_excl, meta_X_excl])
combined_y_excl = np.concatenate([msk_y_excl, meta_y_excl])

print(f"Condition 1 (all features) : {combined_X_all.shape}")
print(f"Condition 2 (excl Age+Meno): {combined_X_excl.shape}")

# ══════════════════════════════════════════════════════
# Model definitions (Exp C final params)
# ══════════════════════════════════════════════════════
MODELS_C = {
    'RF': RandomForestClassifier(
        n_estimators=300, max_depth=4, min_samples_leaf=7,
        class_weight='balanced', criterion='entropy',
        random_state=42, n_jobs=-1),
    'GBM': GradientBoostingClassifier(
        n_estimators=300, max_depth=3, learning_rate=0.01,
        subsample=0.5, criterion='squared_error', loss='exponential',
        random_state=42),
    'AdaBoost': AdaBoostClassifier(
        estimator=DecisionTreeClassifier(max_depth=3, random_state=42),
        n_estimators=200, learning_rate=0.05, random_state=42),
    'XGB': XGBClassifier(
        n_estimators=300, max_depth=3, learning_rate=0.01,
        subsample=0.5, eval_metric='logloss',
        use_label_encoder=False, random_state=42, n_jobs=-1),
}

MODEL_NAMES  = list(MODELS_C.keys())
MODEL_LABELS = ['RF', 'GBM', 'AdaBoost', 'XGBoost']

# ══════════════════════════════════════════════════════
# Run Exp C for both conditions
# ══════════════════════════════════════════════════════
def run_cv_aucs(combined_X, combined_y, model_name, model):
    cv   = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=42)
    aucs = []
    for tr_idx, te_idx in cv.split(combined_X.values, combined_y):
        m = sklearn.base.clone(model)
        m.fit(combined_X.values[tr_idx], combined_y[tr_idx])
        proba = m.predict_proba(combined_X.values[te_idx])[:, 1]
        aucs.append(roc_auc_score(combined_y[te_idx], proba))
    return np.array(aucs)

aucs_all  = {}
aucs_excl = {}

for model_name in MODEL_NAMES:
    print(f"\nRunning {model_name}...")
    aucs_all[model_name]  = run_cv_aucs(combined_X_all,  combined_y_all,
                                         model_name, MODELS_C[model_name])
    aucs_excl[model_name] = run_cv_aucs(combined_X_excl, combined_y_excl,
                                         model_name, MODELS_C[model_name])
    print(f"  All features : {aucs_all[model_name].mean():.3f} ± {aucs_all[model_name].std():.3f}")
    print(f"  Excl Age+Meno: {aucs_excl[model_name].mean():.3f} ± {aucs_excl[model_name].std():.3f}")

# ══════════════════════════════════════════════════════
# Paired dot plot
# ══════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 4, figsize=(18, 7), sharey=True)

color_all  = '#4477AA'
color_excl = '#EE6677'

def pval_label(p):
    if p < 0.001: return '***'
    elif p < 0.01: return '**'
    elif p < 0.05: return '*'
    else: return 'ns'

for i, (model_name, ax) in enumerate(zip(MODEL_NAMES, axes)):
    a = aucs_all[model_name]
    b = aucs_excl[model_name]

    # ── Box plots ──────────────────────────────────────
    ax.boxplot(a, positions=[0], widths=0.40,
               patch_artist=True, showfliers=False,
               medianprops=dict(color='white', linewidth=2.5),
               boxprops=dict(facecolor=color_all, alpha=0.4),
               whiskerprops=dict(color=color_all, linewidth=1.7),
               capprops=dict(color=color_all, linewidth=1.7))

    ax.boxplot(b, positions=[1], widths=0.40,
               patch_artist=True, showfliers=False,
               medianprops=dict(color='white', linewidth=2.5),
               boxprops=dict(facecolor=color_excl, alpha=0.4),
               whiskerprops=dict(color=color_excl, linewidth=1.7),
               capprops=dict(color=color_excl, linewidth=1.7))

    # ── Connecting lines ────────────────────────────────
    jitter = 0.06
    x_a = np.zeros(25) + np.random.uniform(-jitter, jitter, 25)
    x_b = np.ones(25)  + np.random.uniform(-jitter, jitter, 25)
    for j in range(len(a)):
        color_line = '#AAAAAA' if a[j] >= b[j] else '#FFAAAA'
        ax.plot([x_a[j], x_b[j]], [a[j], b[j]],
                color=color_line, linewidth=1, alpha=0.5, zorder=2)

    # ── Scatter dots on top ────────────────────────────
    ax.scatter(x_a, a, color=color_all,  s=50, zorder=3, alpha=0.85,
               edgecolors='white', linewidths=0.3)
    ax.scatter(x_b, b, color=color_excl, s=50, zorder=3, alpha=0.85,
               edgecolors='white', linewidths=0.3)

    # ── P-value ────────────────────────────────────────
    _, p = wilcoxon(a, b)
    label = pval_label(p)
    y_line = max(a.max(), b.max()) + 0.010
    ax.plot([0, 0, 1, 1],
            [y_line, y_line + 0.005, y_line + 0.005, y_line],
            color='black', linewidth=1.4)
    ax.text(0.5, y_line + 0.004, label,
            ha='center', va='bottom', fontsize=20, fontweight='bold')

    # ── Formatting per panel ───────────────────────────
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['All\nfeatures', 'Excl. Age\n& Meno.'], fontsize=19)
    ax.set_title(MODEL_LABELS[i], fontsize=20, fontweight='bold', pad=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='y', labelsize=20)

# ── Shared y axis ──────────────────────────────────────
axes[0].set_ylabel('AUC', fontsize=20, labelpad=10)
axes[0].set_ylim(0.675, None)

# ── Legend ─────────────────────────────────────────────
legend_elements = [
    Patch(facecolor=color_all,  alpha=0.85, label='All features'),
    Patch(facecolor=color_excl, alpha=0.85, label='Excl. Age & Menopausal Status'),
]
fig.legend(handles=legend_elements, fontsize=20, frameon=False,
           loc='lower center', ncol=2, bbox_to_anchor=(0.5, -0.02))

plt.tight_layout(rect=[0, 0.06, 1, 1])

SAVE_DIR = os.path.join(base_dir, 'datasets', 'figures')
os.makedirs(SAVE_DIR, exist_ok=True)
save_path = os.path.join(SAVE_DIR, 'paired_sensitivity_age_meno.png')
plt.savefig(save_path, dpi=450, bbox_inches='tight')
plt.close()

print(f"\nPlot saved → {save_path}")
