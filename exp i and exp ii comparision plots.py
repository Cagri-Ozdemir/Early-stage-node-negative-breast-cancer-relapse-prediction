import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
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
# Load
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

# ── All features ───────────────────────────────────────
DROP_ALL  = ['T_Stage', 'N_Stage']
msk_X_all,  msk_y_all  = load_dataset(MSK_DIR,  CORE_GENES, DROP_ALL)
meta_X_all, meta_y_all = load_dataset(META_DIR, CORE_GENES, DROP_ALL)

# ── Exclude Age + Menopausal ───────────────────────────
DROP_EXCL = ['T_Stage', 'N_Stage', 'Menopausal_Status', 'Age']
msk_X_excl,  msk_y_excl  = load_dataset(MSK_DIR,  CORE_GENES, DROP_EXCL)
meta_X_excl, meta_y_excl = load_dataset(META_DIR, CORE_GENES, DROP_EXCL)

# ══════════════════════════════════════════════════════
# Model definitions
# ══════════════════════════════════════════════════════
MODELS_A = {
    'RF': RandomForestClassifier(
        n_estimators=100, max_depth=4, min_samples_leaf=3,
        class_weight='balanced_subsample', criterion='entropy',
        random_state=42, n_jobs=-1),
    'GBM': GradientBoostingClassifier(
        n_estimators=100, max_depth=3, learning_rate=0.05,
        subsample=0.7, criterion='squared_error', loss='log_loss',
        random_state=42),
    'AdaBoost': AdaBoostClassifier(
        estimator=DecisionTreeClassifier(max_depth=3, random_state=42),
        n_estimators=300, learning_rate=0.05, random_state=42),
    'XGB': XGBClassifier(
        n_estimators=200, max_depth=3, learning_rate=0.05,
        subsample=0.9, eval_metric='logloss',
        use_label_encoder=False, random_state=42, n_jobs=-1),
}

MODELS_B = {
    'RF': RandomForestClassifier(
        n_estimators=200, max_depth=4, min_samples_leaf=3,
        class_weight='balanced_subsample', criterion='gini',
        random_state=42, n_jobs=-1),
    'GBM': GradientBoostingClassifier(
        n_estimators=100, max_depth=4, learning_rate=0.03,
        subsample=0.5, criterion='squared_error', loss='exponential',
        random_state=42),
    'AdaBoost': AdaBoostClassifier(
        estimator=DecisionTreeClassifier(max_depth=3, random_state=42),
        n_estimators=200, learning_rate=0.01, random_state=42),
    'XGB': XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.01,
        subsample=0.7, eval_metric='logloss',
        use_label_encoder=False, random_state=42, n_jobs=-1),
}

MODEL_NAMES  = list(MODELS_A.keys())
MODEL_LABELS = ['RF', 'GBM', 'AdaBoost', 'XGBoost']

# ══════════════════════════════════════════════════════
# Run Exp A and B for both conditions
# ══════════════════════════════════════════════════════
auc_A_all, auc_A_excl = {}, {}
auc_B_all, auc_B_excl = {}, {}
bac_A_all, bac_A_excl = {}, {}
bac_B_all, bac_B_excl = {}, {}

for model_name in MODEL_NAMES:
    print(f"\n── {model_name} ──────────────────")

    # Exp A — all features
    m = sklearn.base.clone(MODELS_A[model_name])
    m.fit(msk_X_all.values, msk_y_all)
    auc_A_all[model_name] = roc_auc_score(meta_y_all, m.predict_proba(meta_X_all.values)[:, 1])
    bac_A_all[model_name] = balanced_accuracy_score(meta_y_all, m.predict(meta_X_all.values))

    # Exp A — excl Age + Meno
    m = sklearn.base.clone(MODELS_A[model_name])
    m.fit(msk_X_excl.values, msk_y_excl)
    auc_A_excl[model_name] = roc_auc_score(meta_y_excl, m.predict_proba(meta_X_excl.values)[:, 1])
    bac_A_excl[model_name] = balanced_accuracy_score(meta_y_excl, m.predict(meta_X_excl.values))

    # Exp B — all features
    m = sklearn.base.clone(MODELS_B[model_name])
    m.fit(meta_X_all.values, meta_y_all)
    auc_B_all[model_name] = roc_auc_score(msk_y_all, m.predict_proba(msk_X_all.values)[:, 1])
    bac_B_all[model_name] = balanced_accuracy_score(msk_y_all, m.predict(msk_X_all.values))

    # Exp B — excl Age + Meno
    m = sklearn.base.clone(MODELS_B[model_name])
    m.fit(meta_X_excl.values, meta_y_excl)
    auc_B_excl[model_name] = roc_auc_score(msk_y_excl, m.predict_proba(msk_X_excl.values)[:, 1])
    bac_B_excl[model_name] = balanced_accuracy_score(msk_y_excl, m.predict(msk_X_excl.values))

    print(f"  Exp A — all: {auc_A_all[model_name]:.3f}  excl: {auc_A_excl[model_name]:.3f}")
    print(f"  Exp B — all: {auc_B_all[model_name]:.3f}  excl: {auc_B_excl[model_name]:.3f}")

# ══════════════════════════════════════════════════════
# Plot helper
# ══════════════════════════════════════════════════════
FONT = 22
color_all  = '#4477AA'
color_excl = '#EE6677'
SAVE_DIR   = os.path.join(base_dir, 'datasets', 'figures')
os.makedirs(SAVE_DIR, exist_ok=True)

def plot_comparison(auc_all, auc_excl, ylabel, save_path):
    fig, ax = plt.subplots(figsize=(10, 6))

    x     = np.arange(len(MODEL_NAMES))
    width = 0.30

    ax.bar(x - width/2, [auc_all[m]  for m in MODEL_NAMES],
           width, color=color_all,  label='All features',
           edgecolor='white', linewidth=0.5)

    ax.bar(x + width/2, [auc_excl[m] for m in MODEL_NAMES],
           width, color=color_excl, label='Excl. Age & Menopausal Status',
           edgecolor='white', linewidth=0.5)

    ax.set_xlabel('Models',  fontsize=FONT, labelpad=10)
    ax.set_ylabel(ylabel,    fontsize=FONT, labelpad=10)
    ax.set_xticks(x)
    ax.set_xticklabels(MODEL_LABELS, fontsize=FONT)
    ax.tick_params(axis='both', labelsize=FONT)
    ax.set_ylim(0.50, 0.95)
    ax.yaxis.set_major_locator(plt.MultipleLocator(0.05))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(fontsize=FONT, frameon=False, loc='upper right')

    plt.tight_layout()
    plt.savefig(save_path, dpi=450, bbox_inches='tight')
    plt.close()
    print(f"Saved → {save_path}")

# ── Exp A — AUC ───────────────────────────────────────
plot_comparison(
    auc_A_all, auc_A_excl,
    ylabel    = 'AUC',
    save_path = os.path.join(SAVE_DIR, 'auc_expA_comparison.png')
)

# ── Exp A — Balanced Accuracy ─────────────────────────
plot_comparison(
    bac_A_all, bac_A_excl,
    ylabel    = 'Balanced Accuracy',
    save_path = os.path.join(SAVE_DIR, 'bac_expA_comparison.png')
)

# ── Exp B — AUC ───────────────────────────────────────
plot_comparison(
    auc_B_all, auc_B_excl,
    ylabel    = 'AUC',
    save_path = os.path.join(SAVE_DIR, 'auc_expB_comparison.png')
)

# ── Exp B — Balanced Accuracy ─────────────────────────
plot_comparison(
    bac_B_all, bac_B_excl,
    ylabel    = 'Balanced Accuracy',
    save_path = os.path.join(SAVE_DIR, 'bac_expB_comparison.png')
)

print(f"\n── Done ──────────────────────────────────────────────")
