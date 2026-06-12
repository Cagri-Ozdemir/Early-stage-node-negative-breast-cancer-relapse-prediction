import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.metrics import (
    roc_auc_score, accuracy_score, f1_score,
    precision_score, recall_score, confusion_matrix,
    balanced_accuracy_score, average_precision_score
)
from sklearn.model_selection import RepeatedStratifiedKFold
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

CORE_GENES = ['TP53', 'PIK3CA']  # for SNV only
DROP_CLIN  = ['T_Stage', 'N_Stage', 'Age', 'Menopausal_Status']

def load_dataset(path, genes):
    snv  = pd.read_csv(os.path.join(path, 'snv_data.csv'),          index_col=0)
    cadd = pd.read_csv(os.path.join(path, 'cadd_features.csv'),     index_col=0)
    clin = pd.read_csv(os.path.join(path, 'clinical_features.csv'), index_col=0)
    lab  = pd.read_csv(os.path.join(path, 'response.csv'),          index_col=0)
    cna  = pd.read_csv(os.path.join(path, 'cna_features.csv'),      index_col=0)

    clin      = clin.drop(columns=DROP_CLIN, errors='ignore')
    avail_snv = [g for g in genes if g in snv.columns]
    cadd_cols = [c for c in cadd.columns]          # ← all cols in file
    raw_cols  = [c for c in cna.columns            # ← all cols in file
                 if c.endswith('_cna_raw')]

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

print(f"MSK      : {msk_X.shape}  | relapse={msk_y.sum()}  non-relapse={(msk_y==0).sum()}")
print(f"METABRIC : {meta_X.shape}  | relapse={meta_y.sum()}  non-relapse={(meta_y==0).sum()}")

combined_X = pd.concat([msk_X, meta_X])
combined_y = np.concatenate([msk_y, meta_y])

# ══════════════════════════════════════════════════════
# Model definitions
# ══════════════════════════════════════════════════════
MODELS = {
    'RF': RandomForestClassifier(
        n_estimators=200, min_samples_leaf=5,
        max_depth=3,
        class_weight='balanced',
        criterion='entropy',
        random_state=42, n_jobs=-1
    ),
    'GBM': GradientBoostingClassifier(
        n_estimators=300, learning_rate=0.01,
        max_depth=2, subsample=0.9,
        loss='exponential',
        random_state=42
    ),
    'AdaBoost': AdaBoostClassifier(
        estimator=DecisionTreeClassifier(
            max_depth=2,
            random_state=42
        ),
        n_estimators=300,
        learning_rate=0.01,
        random_state=42
    ),
    'XGB': XGBClassifier(
        n_estimators=300, learning_rate=0.01,
        max_depth=2, subsample=0.9,
        eval_metric='auc',
        use_label_encoder=False,
        random_state=42, n_jobs=-1
    ),
}

# ══════════════════════════════════════════════════════
# Metrics helpers
# ══════════════════════════════════════════════════════
def print_metrics(y_true, y_pred, y_prob, label=''):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    print(f"\n{'='*60}\n── {label}\n{'='*60}")
    print(f"\n  ── Relapse (class 1) ──────────────────────────")
    print(f"  AUC         : {roc_auc_score(y_true, y_prob):.3f}")
    print(f"  AUCPR       : {average_precision_score(y_true, y_prob):.3f}")
    print(f"  Accuracy    : {accuracy_score(y_true, y_pred):.3f}")
    print(f"  F1          : {f1_score(y_true, y_pred, pos_label=1, zero_division=0):.3f}")
    print(f"  Recall      : {recall_score(y_true, y_pred, pos_label=1):.3f}")
    print(f"  Precision   : {precision_score(y_true, y_pred, pos_label=1, zero_division=0):.3f}")
    print(f"\n  ── Non-Relapse (class 0) ──────────────────────")
    print(f"  Specificity : {tn/(tn+fp):.3f}")
    print(f"  F1          : {f1_score(y_true, y_pred, pos_label=0, zero_division=0):.3f}")
    print(f"\n  ── Confusion Matrix ───────────────────────────")
    print(f"  TP={tp}  FP={fp}")
    print(f"  FN={fn}  TN={tn}")
    print(f"  Relapse correctly identified  : {tp}/{tp+fn} ({tp/(tp+fn):.3f})")
    print(f"  Non-relapse correctly cleared : {tn}/{tn+fp} ({tn/(tn+fp):.3f})")
    print(f"  Prob gap : {y_prob[y_true==1].mean()-y_prob[y_true==0].mean():.3f}")
    return roc_auc_score(y_true, y_prob)


def run_cross_cohort(X_tr, y_tr, X_te, y_te, model_name, model, label=''):
    m = sklearn.base.clone(model)
    m.fit(X_tr, y_tr)
    proba = m.predict_proba(X_te)[:, 1]
    pred  = m.predict(X_te)
    auc = print_metrics(y_te, pred, proba, label=f'[{model_name}] {label}')
    return auc, proba, pred


def run_cv(X, y, model_name, model, label='', n_splits=5, n_repeats=5):
    cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats,
                                 random_state=42)
    aucs, accs, f1s, recs, precs, auprs = [], [], [], [], [], []
    specs, prec_0s, f1_0s               = [], [], []
    tps, fps, fns, tns                  = [], [], [], []

    for tr_idx, te_idx in cv.split(X, y):
        X_tr, X_te = X[tr_idx], X[te_idx]
        y_tr, y_te = y[tr_idx], y[te_idx]

        m = sklearn.base.clone(model)
        m.fit(X_tr, y_tr)
        proba = m.predict_proba(X_te)[:, 1]
        pred  = m.predict(X_te)

        tn, fp, fn, tp = confusion_matrix(y_te, pred).ravel()
        tps.append(tp); fps.append(fp); fns.append(fn); tns.append(tn)
        aucs.append(roc_auc_score(y_te, proba))
        accs.append(balanced_accuracy_score(y_te, pred))
        f1s.append(f1_score(y_te, pred,   pos_label=1, zero_division=0))
        recs.append(recall_score(y_te, pred, pos_label=1))
        precs.append(precision_score(y_te, pred, pos_label=1, zero_division=0))
        auprs.append(average_precision_score(y_te, proba))
        specs.append(tn / (tn + fp))
        prec_0s.append(tn / (tn + fn) if (tn + fn) > 0 else 0.0)
        f1_0s.append(f1_score(y_te, pred, pos_label=0, zero_division=0))

    def s(lst): a = np.array(lst); return a.mean(), a.std()

    print(f"\n{'='*60}\n── [{model_name}] {label}\n{'='*60}")
    print(f"Evaluations      : {len(aucs)} ({n_splits}-fold x {n_repeats} repeats)")
    print(f"\n  ── Relapse (class 1) ─────────────────────────")
    print(f"  AUC            : {s(aucs)[0]:.3f} ± {s(aucs)[1]:.3f}")
    print(f"  AUCPR          : {s(auprs)[0]:.3f} ± {s(auprs)[1]:.3f}")
    print(f"  F1             : {s(f1s)[0]:.3f}  ± {s(f1s)[1]:.3f}")
    print(f"  Recall         : {s(recs)[0]:.3f} ± {s(recs)[1]:.3f}")
    print(f"  Precision      : {s(precs)[0]:.3f} ± {s(precs)[1]:.3f}")
    print(f"\n  ── Non-Relapse (class 0) ──────────────────────")
    print(f"  Specificity    : {s(specs)[0]:.3f} ± {s(specs)[1]:.3f}")
    print(f"  Precision      : {s(prec_0s)[0]:.3f} ± {s(prec_0s)[1]:.3f}")
    print(f"  F1             : {s(f1_0s)[0]:.3f} ± {s(f1_0s)[1]:.3f}")
    print(f"\n  ── Confusion Matrix (mean per fold) ───────────")
    print(f"  TP={np.mean(tps):.1f}  FP={np.mean(fps):.1f}")
    print(f"  FN={np.mean(fns):.1f}  TN={np.mean(tns):.1f}")

    return s(aucs)[0]

# ══════════════════════════════════════════════════════
# Run all experiments for all models
# ══════════════════════════════════════════════════════
summary = {m: {} for m in MODELS}

for model_name, model in MODELS.items():
    print(f"\n{'#'*70}")
    print(f"## MODEL: {model_name}")
    print(f"{'#'*70}")

    # ── Exp A: Train MSK → Test METABRIC ──────────────
    auc_A, _, _ = run_cross_cohort(
        msk_X.values, msk_y,
        meta_X.values, meta_y,
        model_name, model,
        f'Exp A: Train MSK ({len(msk_X)}) → Test METABRIC ({len(meta_X)})'
    )
    summary[model_name]['Exp_A'] = auc_A

    # ── Exp B: Train METABRIC → Test MSK ──────────────
    auc_B, _, _ = run_cross_cohort(
        meta_X.values, meta_y,
        msk_X.values, msk_y,
        model_name, model,
        f'Exp B: Train METABRIC ({len(meta_X)}) → Test MSK ({len(msk_X)})'
    )
    summary[model_name]['Exp_B'] = auc_B

    # ── Exp C: MSK + METABRIC Combined CV (5-fold x 5) ─
    auc_C = run_cv(
        combined_X.values, combined_y,
        model_name, model,
        f'Exp C: MSK+METABRIC Combined CV ({len(combined_X)} samples)',
        n_splits=5, n_repeats=5
    )
    summary[model_name]['Exp_C'] = auc_C

# ══════════════════════════════════════════════════════
# Summary table
# ══════════════════════════════════════════════════════
print(f"\n{'='*85}")
print(f"── AUC Summary — All Models")
print(f"{'='*85}")

exps = ['Exp_A', 'Exp_B', 'Exp_C']
labels_exp = [
    'A: MSK → METABRIC',
    'B: METABRIC → MSK',
    'C: MSK+METABRIC CV (5x5)',
]

print(f"{'Experiment':<30}", end='')
for m in MODELS:
    print(f"  {m:>8}", end='')
print()
print(f"{'-'*85}")
for exp, lbl in zip(exps, labels_exp):
    print(f"  {lbl:<28}", end='')
    for m in MODELS:
        val = summary[m].get(exp, 0)
        print(f"  {val:>8.3f}", end='')
    print()
print(f"{'='*85}")


print(f"\n── Done ──────────────────────────────────────────────")
