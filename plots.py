import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.metrics import (
    roc_auc_score, confusion_matrix, average_precision_score
)
from sklearn.model_selection import RepeatedStratifiedKFold
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
import sklearn.base
import warnings
from sklearn.tree import DecisionTreeClassifier
warnings.filterwarnings('ignore')

base_dir = '/Users/cagriozdemir/PycharmProjects/Machine learning-based relapse prediction '
MSK_DIR  = os.path.join(base_dir, 'datasets', 'MSK_data')
META_DIR = os.path.join(base_dir, 'datasets', 'METABRIC_data')

CORE_GENES = ['TP53', 'PIK3CA']
DROP_CLIN  = ['T_Stage', 'N_Stage']

# ══════════════════════════════════════════════════════
# Load
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

msk_X,  msk_y  = load_dataset(MSK_DIR,  CORE_GENES)
meta_X, meta_y = load_dataset(META_DIR, CORE_GENES)
combined_X     = pd.concat([msk_X, meta_X])
combined_y     = np.concatenate([msk_y, meta_y])

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

MODEL_NAMES = list(MODELS_A.keys())

# ══════════════════════════════════════════════════════
# Run experiments and collect AUC
# ══════════════════════════════════════════════════════
results = {m: {'A': None, 'B': None, 'C_mean': None, 'C_std': None}
           for m in MODEL_NAMES}

for model_name in MODEL_NAMES:
    print(f"\n── {model_name} ──────────────────────────────────")

    # Exp A: Train MSK → Test METABRIC
    m = sklearn.base.clone(MODELS_A[model_name])
    m.fit(msk_X.values, msk_y)
    proba = m.predict_proba(meta_X.values)[:, 1]
    results[model_name]['A'] = roc_auc_score(meta_y, proba)
    print(f"  Exp A AUC: {results[model_name]['A']:.3f}")

    # Exp B: Train METABRIC → Test MSK
    m = sklearn.base.clone(MODELS_B[model_name])
    m.fit(meta_X.values, meta_y)
    proba = m.predict_proba(msk_X.values)[:, 1]
    results[model_name]['B'] = roc_auc_score(msk_y, proba)
    print(f"  Exp B AUC: {results[model_name]['B']:.3f}")

    # Exp C: Combined CV — collect all 25 AUC values
    cv   = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=42)
    aucs = []
    for tr_idx, te_idx in cv.split(combined_X.values, combined_y):
        m = sklearn.base.clone(MODELS_C[model_name])
        m.fit(combined_X.values[tr_idx], combined_y[tr_idx])
        proba = m.predict_proba(combined_X.values[te_idx])[:, 1]
        aucs.append(roc_auc_score(combined_y[te_idx], proba))
    results[model_name]['C_mean'] = np.mean(aucs)
    results[model_name]['C_std']  = np.std(aucs)
    print(f"  Exp C AUC: {results[model_name]['C_mean']:.3f} ± {results[model_name]['C_std']:.3f}")

# ══════════════════════════════════════════════════════
# Plot
# ══════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(10, 6))

x         = np.arange(len(MODEL_NAMES))
width     = 0.25
x_labels  = ['RF', 'GBM', 'AdaBoost', 'XGBoost']
colors = {'A': '#AED6F1', 'B': '#2E86C1', 'C': '#1A5276'}
bars_A = ax.bar(x - width, [results[m]['A']      for m in MODEL_NAMES],
                width, color=colors['A'], label='Exp (i): MSK → METABRIC',
                edgecolor='white', linewidth=0.5)

bars_B = ax.bar(x,          [results[m]['B']      for m in MODEL_NAMES],
                width, color=colors['B'], label='Exp (ii): METABRIC → MSK',
                edgecolor='white', linewidth=0.5)

bars_C = ax.bar(x + width,  [results[m]['C_mean'] for m in MODEL_NAMES],
                width, color=colors['C'], label='Exp (iii): Combined CV',
                edgecolor='white', linewidth=0.5,
                yerr=[results[m]['C_std'] for m in MODEL_NAMES],
                capsize=4, error_kw=dict(ecolor='gray', elinewidth=1.2, capthick=1.2))

# ── Formatting ─────────────────────────────────────
ax.set_xlabel('Models', fontsize=20, labelpad=10)
ax.set_ylabel('AUC', fontsize=20, labelpad=10)
ax.set_xticks(x)
ax.set_xticklabels(x_labels, fontsize=17)
ax.set_ylim(0.5, 0.95)
ax.yaxis.set_major_locator(plt.MultipleLocator(0.05))
ax.tick_params(axis='both', labelsize=18)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.legend(fontsize=17, frameon=False, loc='upper right')

plt.tight_layout()

SAVE_DIR = os.path.join(base_dir, 'datasets', 'figures')
os.makedirs(SAVE_DIR, exist_ok=True)
save_path = os.path.join(SAVE_DIR, 'auc_performance.png')
plt.savefig(save_path, dpi=450, bbox_inches='tight')
plt.close()

print(f"\nPlot saved → {save_path}")

