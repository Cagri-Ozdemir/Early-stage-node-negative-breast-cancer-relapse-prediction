import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import shap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import joblib
import warnings
warnings.filterwarnings('ignore')

base_dir = '/Users/cagriozdemir/PycharmProjects/Machine learning-based relapse prediction '
MSK_DIR  = os.path.join(base_dir, 'datasets', 'MSK_data')
META_DIR = os.path.join(base_dir, 'datasets', 'METABRIC_data')
MODEL_DIR = os.path.join(base_dir, 'datasets', 'final_model')
os.makedirs(MODEL_DIR, exist_ok=True)

CORE_GENES = ['TP53', 'PIK3CA']
DROP_CLIN  = ['T_Stage', 'N_Stage']

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
# Load training data
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

# Rename for display
combined_X_renamed = combined_X.rename(columns=RENAME_MAP)
FEATURE_NAMES      = combined_X_renamed.columns.tolist()

print(f"Training on combined cohort: {combined_X_renamed.shape}")
print(f"Features: {FEATURE_NAMES}")

# ══════════════════════════════════════════════════════
# Train final model on full combined cohort
# ══════════════════════════════════════════════════════
print(f"\n── Training final RF model on full combined cohort ────────")
rf_final = RandomForestClassifier(**RF_PARAMS)
rf_final.fit(combined_X_renamed.values, combined_y)
print(f"  Model trained on {len(combined_y)} patients ✓")

# ── Save model and feature names ───────────────────────
joblib.dump(rf_final,     os.path.join(MODEL_DIR, 'rf_final_model.joblib'))
joblib.dump(FEATURE_NAMES, os.path.join(MODEL_DIR, 'feature_names.joblib'))
print(f"  Model saved to: {MODEL_DIR}")

# ── Fit SHAP explainer ────────────────────────────────
explainer = shap.TreeExplainer(rf_final)
joblib.dump(explainer, os.path.join(MODEL_DIR, 'shap_explainer.joblib'))
print(f"  SHAP explainer saved ✓")

# ══════════════════════════════════════════════════════
# Prediction function for new patients
# ══════════════════════════════════════════════════════
def predict_patient(patient_data: dict,
                    model_dir: str = MODEL_DIR,
                    save_plot: str = None,
                    verbose: bool  = True):
    """
    Predict relapse risk for a new patient.

    Parameters
    ----------
    patient_data : dict
        Dictionary with feature values. Keys must match FEATURE_NAMES.
        Example:
        {
            'FGA'              : 0.35,
            'TMB'              : 0.12,
            'Age'              : 0.52,
            'Grade'            : 3,
            'ER Status'        : 1,
            'PR Status'        : 1,
            'HER2 Status'      : 0,
            'Menopausal Status': 1,
            'TP53 SNVs'        : 1,
            'TP53 CADD'        : 25.3,
            'PIK3CA SNVs'      : 0,
            'PIK3CA CADD'      : 0.0,
            'CCND1 CNAs'       : 0.5,
            'MDM4 CNAs'        : 0.0,
        }
    model_dir  : str  — path to saved model directory
    save_plot  : str  — path to save SHAP waterfall plot (optional)
    verbose    : bool — print results to console

    Returns
    -------
    dict with keys: risk_probability, risk_group, shap_values
    """

    # Load model and explainer
    model         = joblib.load(os.path.join(model_dir, 'rf_final_model.joblib'))
    feature_names = joblib.load(os.path.join(model_dir, 'feature_names.joblib'))
    explainer     = joblib.load(os.path.join(model_dir, 'shap_explainer.joblib'))

    # Build input array in correct feature order
    missing = [f for f in feature_names if f not in patient_data]
    if missing:
        print(f"WARNING: Missing features set to 0: {missing}")

    x = np.array([[patient_data.get(f, 0) for f in feature_names]])

    # Predict
    prob       = model.predict_proba(x)[0, 1]
    risk_group = 'High Risk' if prob >= 0.5 else 'Low Risk'

    # SHAP values — RF returns list [class0, class1]
    shap_vals = explainer.shap_values(x)

    if isinstance(shap_vals, list):
        sv           = shap_vals[1][0]
        expected_val = explainer.expected_value[1]
    elif hasattr(shap_vals, 'ndim') and shap_vals.ndim == 3:
        sv           = shap_vals[0, :, 1]
        expected_val = (explainer.expected_value[1]
                        if hasattr(explainer.expected_value, '__len__')
                        else explainer.expected_value)
    else:
        sv           = shap_vals[0]
        expected_val = (explainer.expected_value[1]
                        if hasattr(explainer.expected_value, '__len__')
                        else explainer.expected_value)

    # Sanity check
    print(f"\n  Expected value (base): {expected_val:.4f}")
    print(f"  SHAP sum + base = {sv.sum() + expected_val:.4f} (should be close to {prob:.4f})")

    # ── SHAP waterfall plot ────────────────────────────

    shap_exp = shap.Explanation(
        values        = sv,
        base_values   = expected_val,
        data          = x[0],
        feature_names = feature_names
    )

    # ── Custom decision plot with clear feature values and dots ──
    sorted_idx  = np.argsort(np.abs(sv))  # ascending — bottom to top
    sorted_feat = [feature_names[i] for i in sorted_idx]
    sorted_sv   = sv[sorted_idx]
    sorted_data = x[0][sorted_idx]

    # Build cumulative path starting exactly from 0.5
    # Adjust sv so sum starts at 0.5 (offset by difference from expected_val)
    offset     = 0.5 - expected_val
    adj_sv     = sorted_sv + offset / len(sorted_sv)
    cumulative = [0.5]
    for val in adj_sv:
        cumulative.append(cumulative[-1] + val)

    fig, ax = plt.subplots(figsize=(12, 9))

    # ── Draw horizontal lines per feature ──────────────
    for i, (feat, fval, dval, afval) in enumerate(zip(sorted_feat, sorted_sv, sorted_data, adj_sv)):
        x_start = cumulative[i]
        x_end   = cumulative[i + 1]
        color   = '#E74C3C' if fval >= 0 else '#2E86C1'

        # Horizontal line
        ax.plot([x_start, x_end], [i, i], color=color, linewidth=2.5, zorder=2)

        # Start dot (small grey)
        ax.scatter(x_start, i, color='grey', s=40, zorder=3)

        # End dot (large colored)
        ax.scatter(x_end, i, color=color, s=120, zorder=4, edgecolors='white', linewidths=1.5)

        # Feature value label on left
        ax.text(-0.005, i, f'{dval} = {feat}',
                transform=ax.get_yaxis_transform(),
                ha='right', va='center', fontsize=15,
                color='#555555')

        # SHAP value label — right of dot if positive, left if negative
        if abs(fval) >= 0.001:
            if fval >= 0:
                ax.text(x_end + 0.009, i, f'{fval:+.3f}',
                        ha='left', va='center', fontsize=15,
                        color=color, fontweight='bold')
            else:
                ax.text(x_end - 0.009, i, f'{fval:+.3f}',
                        ha='right', va='center', fontsize=15,
                        color=color, fontweight='bold')
        color = '#E74C3C' if afval >= 0 else '#2E86C1'

    # ── Vertical dashed line at base value 0.5 ─────────
    ax.axvline(x=0.5, color='gray', linestyle='--', linewidth=1.3, alpha=0.7)
    ax.text(0.5, -0.8, '0.5', ha='center', va='top', fontsize=15,
            color='gray', fontweight='bold')

    # ── Final prediction line ───────────────────────────
    pred_color = '#E74C3C' if prob >= 0.5 else '#2E86C1'
    ax.axvline(x=cumulative[-1], color=pred_color, linestyle='-', linewidth=1.5, alpha=0.5)
    ax.text(cumulative[-1] + 0.003, len(sorted_feat) - 0.3,
            f'f(x) = {cumulative[-1]:.3f}',
            fontsize=15, fontweight='bold', color=pred_color)

    # ── Formatting ──────────────────────────────────────
    ax.set_yticks([])
    ax.set_xlabel('Relapse Risk Probability', fontsize=18, labelpad=10)
    ax.set_xlim(0.0, 1)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.tick_params(axis='x', labelsize=15)

    plt.tight_layout()

    if save_plot:
        plt.savefig(save_plot, dpi=450, bbox_inches='tight')
        print(f"  SHAP waterfall plot saved → {save_plot}")
    else:
        default_path = os.path.join(MODEL_DIR, 'patient_shap_waterfall.png')
        plt.savefig(default_path, dpi=450, bbox_inches='tight')
        print(f"  SHAP waterfall plot saved → {default_path}")
    plt.close()

    if verbose:
        print(f"\n{'='*55}")
        print(f"── Relapse Risk Prediction")
        print(f"{'='*55}")
        print(f"  Risk Probability : {prob:.3f}")
        print(f"  Risk Group       : {risk_group}")
        print(f"\n  Top contributing features (|SHAP|):")
        top_idx = np.argsort(np.abs(sv))[::-1][:5]
        for i in top_idx:
            direction = '↑ relapse' if sv[i] > 0 else '↓ relapse'
            print(f"    {feature_names[i]:<22} SHAP={sv[i]:+.4f}  ({direction})")
        print(f"{'='*55}")

    return {
        'risk_probability' : prob,
        'risk_group'       : risk_group,
        'shap_values'      : dict(zip(feature_names, sv))
    }

# ══════════════════════════════════════════════════════
# Example — predict for a single patient
#══════════════════════════════════════════════════════
# example_patient = {
#     'FGA'              : 0.35,
#     'TMB'              : 0.20,
#     'Age'              : 0.48,
#     'Grade'            : 3,
#     'ER Status'        : 1,
#     'PR Status'        : 1,
#     'HER2 Status'      : 0,
#     'Menopausal Status': 0,
#     'TP53 SNVs'        : 1,
#     'TP53 CADD'        : 28.5,
#     'PIK3CA SNVs'      : 0,
#     'PIK3CA CADD'      : 0.0,
#     'CCND1 CNAs'       : 0.8,
#     'MDM4 CNAs'        : 0.0,
# }
example_patient = {
    'FGA'              : 0.05,
    'TMB'              : -0.30,
    'Age'              : 0.70,
    'Grade'            : 1,
    'ER Status'        : 1,
    'PR Status'        : 1,
    'HER2 Status'      : 0,
    'Menopausal Status': 1,
    'TP53 SNVs'        : 0,
    'TP53 CADD'        : 0.0,
    'PIK3CA SNVs'      : 1,
    'PIK3CA CADD'      : 15.0,
    'CCND1 CNAs'       : 0.0,
    'MDM4 CNAs'        : 0.0,
}
print(f"\n── Example patient prediction ──────────────────────────────")
result = predict_patient(
    patient_data = example_patient,
    save_plot    = os.path.join(MODEL_DIR, 'example_patient_shap2.png')
)

print(f"\n── Done ──────────────────────────────────────────────────")
print(f"  Model directory: {MODEL_DIR}")
print(f"  Files saved:")
print(f"    rf_final_model.joblib")
print(f"    feature_names.joblib")
print(f"    shap_explainer.joblib")
print(f"    example_patient_shap.png")
