"""
Insurance Fraud Detection System — Flask Backend
=================================================
NO EasyOCR / NO PyTorch / NO pdf2image.

Run:   python app.py
Open:  http://localhost:5000

If you see NumPy errors run:
    pip uninstall numpy -y
    pip install "numpy<2"
    pip install lightgbm shap --upgrade
"""

import os
import json
import warnings
import traceback

import numpy as np
import pandas as pd
import joblib

from flask import Flask, render_template, request, jsonify
from datetime import datetime

warnings.filterwarnings("ignore")

# ── Flask setup ───────────────────────────────────────────────────────────────
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024

UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

BASE      = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE, "models")

# ═════════════════════════════════════════════════════════════════
#  LOAD MODELS — graceful fallback if any model fails
# ═════════════════════════════════════════════════════════════════
print("=" * 55)
print("  Loading Insurance Fraud Detection Models...")
print("=" * 55)

# XGBoost
try:
    xgb_model = joblib.load(os.path.join(MODEL_DIR, "model_xgboost.pkl"))
    print("  ✅ XGBoost loaded")
    XGB_OK = True
except Exception as e:
    print(f"  ❌ XGBoost failed: {e}")
    xgb_model = None
    XGB_OK = False

# LightGBM
try:
    lgbm_model = joblib.load(os.path.join(MODEL_DIR, "model_lightgbm.pkl"))
    print("  ✅ LightGBM loaded")
    LGBM_OK = True
except Exception as e:
    print(f"  ⚠️  LightGBM failed: {e}")
    lgbm_model = None
    LGBM_OK = False

# Random Forest
try:
    rf_model = joblib.load(os.path.join(MODEL_DIR, "model_random_forest.pkl"))
    print("  ✅ Random Forest loaded")
    RF_OK = True
except Exception as e:
    print(f"  ❌ Random Forest failed: {e}")
    rf_model = None
    RF_OK = False

if not any([XGB_OK, LGBM_OK, RF_OK]):
    raise RuntimeError("No models loaded — check your models/ folder")

# Feature columns
with open(os.path.join(MODEL_DIR, "final_feature_columns.json")) as fh:
    FEATURE_COLS = json.load(fh)
print(f"  ✅ Features: {len(FEATURE_COLS)}")

# Metadata
with open(os.path.join(MODEL_DIR, "model_metadata.json")) as fh:
    METADATA = json.load(fh)
print(f"  ✅ AUC={METADATA['val_auc_roc']}  F1={METADATA['val_f1']}")

# Ensemble config
with open(os.path.join(MODEL_DIR, "ensemble_config.json")) as fh:
    ENS_CFG = json.load(fh)

# Population stats from X_val
X_val_raw = pd.read_parquet(os.path.join(MODEL_DIR, "X_val.parquet"))
X_val_raw = X_val_raw[[c for c in FEATURE_COLS if c in X_val_raw.columns]]
POP_MEANS = X_val_raw.mean().to_dict()
POP_STDS  = X_val_raw.std().to_dict()
print("  ✅ Population stats loaded")

# SHAP — load only if XGBoost is available, with safe fallback
SHAP_OK   = False
explainer = None
try:
    import shap
    if XGB_OK:
        explainer = shap.TreeExplainer(xgb_model)
        SHAP_OK   = True
        print("  ✅ SHAP explainer ready")
    else:
        print("  ⚠️  SHAP skipped (XGBoost not loaded)")
except Exception as e:
    print(f"  ⚠️  SHAP failed: {e}")
    print("      Fraud reasons will use rule-based fallback instead.")
    print("      Fix: pip uninstall numpy -y && pip install 'numpy<2' && pip install shap --upgrade")

THRESHOLD = METADATA["best_threshold"]
WEIGHTS   = ENS_CFG.get("weights", [3, 3, 1])

# Adjust weights for available models
available_weights = []
available_models  = []
labels = []
if XGB_OK:
    available_models.append(xgb_model)
    available_weights.append(WEIGHTS[0] if len(WEIGHTS) > 0 else 3)
    labels.append("XGBoost")
if LGBM_OK:
    available_models.append(lgbm_model)
    available_weights.append(WEIGHTS[1] if len(WEIGHTS) > 1 else 3)
    labels.append("LightGBM")
if RF_OK:
    available_models.append(rf_model)
    available_weights.append(WEIGHTS[2] if len(WEIGHTS) > 2 else 1)
    labels.append("Random Forest")

print(f"\n  🚀 Ready | Models: {', '.join(labels)}")
print(f"     Threshold={THRESHOLD} | AUC={METADATA['val_auc_roc']}")
print("=" * 55)


# ═════════════════════════════════════════════════════════════════
#  HELPERS
# ═════════════════════════════════════════════════════════════════

def _set(row, key, val):
    if key in row:
        row[key] = val

def _iv(d, key):
    try:
        return float(d.get(key, POP_MEANS.get(key, 0)) or 0)
    except (TypeError, ValueError):
        return float(POP_MEANS.get(key, 0))


# ═════════════════════════════════════════════════════════════════
#  FEATURE ENGINEERING
# ═════════════════════════════════════════════════════════════════

def build_feature_df(input_dict):
    row = {col: float(POP_MEANS.get(col, 0.0)) for col in FEATURE_COLS}
    eps = 1e-9

    reimb   = max(_iv(input_dict, "ip_avg_reimbursement"),     0)
    stays   = max(_iv(input_dict, "ip_avg_stay_days"),         0)
    claims  = max(_iv(input_dict, "ip_claim_count"),           1)
    pats    = max(_iv(input_dict, "total_unique_patients"),    1)
    chronic = max(_iv(input_dict, "ip_avg_chronic_cond"),      0)
    phys    = max(_iv(input_dict, "ip_unique_attending_phys"), 1)

    # Direct
    _set(row, "ip_avg_reimbursement",     reimb)
    _set(row, "ip_avg_stay_days",         stays)
    _set(row, "ip_claim_count",           claims)
    _set(row, "total_unique_patients",    pats)
    _set(row, "ip_avg_chronic_cond",      chronic)
    _set(row, "ip_unique_attending_phys", phys)

    # Derived base
    _set(row, "ip_total_reimbursement",    reimb * claims)
    _set(row, "ip_total_stay_days",        stays * claims)
    _set(row, "ip_max_stay_days",          stays * 1.5)
    _set(row, "ip_avg_patient_age",        _iv(input_dict, "ip_avg_patient_age") or 68)
    _set(row, "op_claim_count",            claims * 0.6)
    _set(row, "op_unique_patients",        pats   * 0.7)
    _set(row, "op_avg_reimbursement",      reimb  * 0.3)
    _set(row, "ip_unique_operating_phys",  max(1, phys * 0.6))
    _set(row, "ip_max_chronic_cond",       min(11, chronic + 1))
    _set(row, "ip_total_deductible",       reimb * 0.08)
    _set(row, "ip_avg_deductible",         reimb * 0.08)
    _set(row, "op_avg_patient_age",        68)
    _set(row, "ip_avg_annual_ip_reimb",    reimb * 12)
    _set(row, "ip_avg_annual_op_reimb",    reimb * 0.3 * 12)
    _set(row, "op_avg_chronic_cond",       chronic)
    _set(row, "op_total_deductible",       reimb * 0.06)
    _set(row, "op_avg_claim_duration",     stays * 0.4)
    _set(row, "op_unique_attending_phys",  max(1, phys * 0.7))
    _set(row, "op_total_reimbursement",    reimb * 0.3 * claims)
    _set(row, "op_max_reimbursement",      reimb * 0.5)
    _set(row, "ip_deceased_patient_count", max(0, pats * 0.02))

    # Ratio features
    _set(row, "feat_reimb_per_ip_claim",       reimb)
    _set(row, "feat_reimb_per_unique_patient", (reimb * claims) / pats)
    _set(row, "feat_reimb_per_physician",      (reimb * claims) / phys)
    _set(row, "feat_claims_per_patient",       claims / pats)
    _set(row, "feat_op_physician_reuse_rate",  claims / phys)
    _set(row, "feat_stay_days_per_claim",      stays)
    _set(row, "feat_deductible_ratio",         0.08)
    _set(row, "feat_ip_op_patient_ratio",      1.4)
    _set(row, "avg_reimb_per_patient",         (reimb * claims) / pats)
    _set(row, "deductible_reimb_ratio",        0.08)

    flags = sum([
        reimb   > POP_MEANS.get("ip_avg_reimbursement",    0) * 2,
        stays   > POP_MEANS.get("ip_avg_stay_days",        0) * 2,
        claims  > POP_MEANS.get("ip_claim_count",          0) * 2,
        pats    > POP_MEANS.get("total_unique_patients",   0) * 2,
        chronic > 5,
        phys    > POP_MEANS.get("ip_unique_attending_phys",0) * 2,
    ])
    _set(row, "feat_total_flags_triggered", flags)

    # Risk scores
    def pct(val, key, mult=3):
        return min(val / max(POP_MEANS.get(key, 1) * mult, eps), 1.0)

    rp = pct(reimb,   "ip_avg_reimbursement")
    sp = pct(stays,   "ip_avg_stay_days")
    cp = min(chronic / 11, 1.0)
    vp = pct(claims,  "ip_claim_count")
    pp = pct(phys,    "ip_unique_attending_phys")
    composite = (2*rp + 1.5*vp + cp + pp + 1.3*sp) / 6.8

    _set(row, "risk_financial",          rp)
    _set(row, "risk_stay_duration",      sp)
    _set(row, "risk_medical_complexity", cp)
    _set(row, "risk_volume",             vp)
    _set(row, "risk_physician_pattern",  pp)
    _set(row, "risk_composite_score",    composite)

    # Interactions
    _set(row, "interact_volume_x_reimb",              np.log1p(claims) * np.log1p(reimb))
    _set(row, "interact_patients_x_chronic",          np.log1p(pats)   * chronic)
    _set(row, "interact_stay_x_reimb",                stays * np.log1p(reimb))
    _set(row, "interact_physicians_x_patients",       np.log1p(phys) * np.log1p(pats))
    _set(row, "interact_financial_x_physician",       rp * pp)
    _set(row, "interact_volume_x_stay",               vp * sp)
    _set(row, "interact_composite_squared",           composite ** 2)
    _set(row, "interact_low_deductible_x_high_reimb", min(12 * np.log1p(reimb * claims), 1e6))

    # Z-scores
    for col in FEATURE_COLS:
        if col.startswith("zscore_"):
            base = col[len("zscore_"):]
            if base in row:
                mean = POP_MEANS.get(base, 0)
                std  = max(POP_STDS.get(base, 1), eps)
                row[col] = (row[base] - mean) / std

    # Bin features
    for col in FEATURE_COLS:
        if col.startswith("bin_"):
            base = col[len("bin_"):]
            if base in row:
                avg = POP_MEANS.get(base, 0)
                row[col] = min(int((row[base] / max(avg * 0.5, eps)) * 1.5), 4)

    # Flag features
    for col in FEATURE_COLS:
        if col.startswith("flag_") and col != "feat_total_flags_triggered":
            if "very_low" in col:
                row[col] = 1 if row.get("feat_deductible_ratio", 1) < 0.05 else 0
            else:
                base     = col.replace("flag_extreme_", "").replace("flag_", "")
                base_val = row.get(base, 0)
                thresh   = POP_MEANS.get(base, 0) * 2
                row[col] = 1 if base_val > thresh else 0

    df = pd.DataFrame([row])[FEATURE_COLS]
    return df.fillna(0).replace([np.inf, -np.inf], 0)


# ═════════════════════════════════════════════════════════════════
#  PREDICTION
# ═════════════════════════════════════════════════════════════════

def ensemble_predict(feat_df):
    scores = []
    xgb_s = lgbm_s = rf_s = 0.0

    for model, w, label in zip(available_models, available_weights, labels):
        p = float(model.predict_proba(feat_df)[:, 1][0])
        scores.append(p * w)
        if label == "XGBoost":    xgb_s  = round(p, 4)
        if label == "LightGBM":   lgbm_s = round(p, 4)
        if label == "Random Forest": rf_s = round(p, 4)

    p_ens = sum(scores) / sum(available_weights)
    return round(p_ens, 4), xgb_s, lgbm_s, rf_s


def get_shap_reasons(feat_df, input_vals):
    """Returns SHAP reasons if available, rule-based fallback otherwise."""

    def iv(k):
        try: return float(input_vals.get(k, POP_MEANS.get(k, 0)) or 0)
        except: return 0.0

    def rx(k, v):
        return v / max(POP_MEANS.get(k, 1), 1e-9)

    reimb   = iv("ip_avg_reimbursement")
    stays   = iv("ip_avg_stay_days")
    claims  = max(iv("ip_claim_count"), 1)
    pats    = max(iv("total_unique_patients"), 1)
    chronic = iv("ip_avg_chronic_cond")
    phys    = iv("ip_unique_attending_phys")

    TMPLS = {
        "ip_avg_reimbursement"          : f"Avg reimbursement ${reimb:,.0f} is {rx('ip_avg_reimbursement', reimb):.1f}x the population average",
        "ip_total_reimbursement"        : f"Total billing ${reimb*claims:,.0f} is {rx('ip_avg_reimbursement', reimb):.1f}x higher than a typical provider",
        "ip_avg_stay_days"              : f"Hospital stay of {stays:.1f} days is {rx('ip_avg_stay_days', stays):.1f}x the average ({POP_MEANS.get('ip_avg_stay_days',0):.1f} days)",
        "ip_claim_count"                : f"Filed {claims:.0f} inpatient claims — {rx('ip_claim_count', claims):.1f}x more than the average provider",
        "total_unique_patients"         : f"Served {pats:.0f} unique patients — {rx('total_unique_patients', pats):.1f}x a typical provider",
        "ip_avg_chronic_cond"           : f"Avg {chronic:.2f} chronic conditions per patient — possible diagnosis stuffing",
        "ip_unique_attending_phys"      : f"Used {phys:.0f} unique physicians — {rx('ip_unique_attending_phys', phys):.1f}x more than average",
        "risk_composite_score"          : "Overall composite fraud risk score is in the high-risk tier",
        "risk_financial"                : "Financial anomaly score is unusually high across billing metrics",
        "risk_volume"                   : "Claim volume risk score is in the top tier",
        "risk_stay_duration"            : "Hospital stay duration is statistically anomalous",
        "risk_physician_pattern"        : "Physician usage pattern is statistically anomalous",
        "feat_reimb_per_unique_patient" : f"Extracts ${reimb*claims/pats:,.0f} per patient — very high per-patient billing",
        "feat_claims_per_patient"       : f"Files {claims/pats:.2f} claims per patient — possible duplicate billing",
        "feat_op_physician_reuse_rate"  : f"Same physicians reused across {claims/phys:.1f} claims on average",
        "feat_total_flags_triggered"    : "Provider triggered multiple extreme-outlier anomaly flags simultaneously",
        "interact_volume_x_reimb"       : "High claim volume combined with high reimbursement — billing mill pattern",
        "interact_stay_x_reimb"         : "Long hospital stay combined with high claim amount — overbilling signal",
        "interact_patients_x_chronic"   : "Large patient base with inflated chronic condition scores — diagnosis stuffing at scale",
        "interact_composite_squared"    : "Composite risk score is extremely elevated — multiple fraud signals amplifying each other",
    }

    # ── SHAP path (best) ──────────────────────────────────────────────────────
    if SHAP_OK and explainer is not None:
        try:
            import shap as shap_lib
            sv     = explainer(feat_df)
            shap_s = pd.Series(sv.values[0], index=FEATURE_COLS)
            top5   = shap_s[shap_s > 0].sort_values(ascending=False).head(5)
            reasons = []
            for feat, sv_val in top5.items():
                reasons.append({
                    "feature": feat,
                    "shap"   : round(float(sv_val), 4),
                    "text"   : TMPLS.get(feat, feat.replace("_", " ").title()),
                })
            if reasons:
                return reasons
        except Exception:
            pass  # fall through to rule-based

    # ── Rule-based fallback (when SHAP unavailable) ───────────────────────────
    candidates = []

    reimb_avg = POP_MEANS.get("ip_avg_reimbursement", 1)
    stay_avg  = POP_MEANS.get("ip_avg_stay_days",      1)
    claim_avg = POP_MEANS.get("ip_claim_count",         1)
    pat_avg   = POP_MEANS.get("total_unique_patients",  1)
    phys_avg  = POP_MEANS.get("ip_unique_attending_phys", 1)

    if reimb  > reimb_avg * 1.5: candidates.append({"feature": "ip_avg_reimbursement",     "shap": round((reimb/reimb_avg-1)*0.1,4), "text": TMPLS["ip_avg_reimbursement"]})
    if stays  > stay_avg  * 1.5: candidates.append({"feature": "ip_avg_stay_days",         "shap": round((stays/stay_avg-1)*0.08,4), "text": TMPLS["ip_avg_stay_days"]})
    if claims > claim_avg * 1.5: candidates.append({"feature": "ip_claim_count",           "shap": round((claims/claim_avg-1)*0.07,4),"text": TMPLS["ip_claim_count"]})
    if pats   > pat_avg   * 1.5: candidates.append({"feature": "total_unique_patients",    "shap": round((pats/pat_avg-1)*0.06,4),   "text": TMPLS["total_unique_patients"]})
    if chronic > 4:              candidates.append({"feature": "ip_avg_chronic_cond",      "shap": round(chronic*0.05,4),            "text": TMPLS["ip_avg_chronic_cond"]})
    if phys   > phys_avg  * 1.5: candidates.append({"feature": "ip_unique_attending_phys", "shap": round((phys/phys_avg-1)*0.05,4), "text": TMPLS["ip_unique_attending_phys"]})

    # Always add composite risk if high
    candidates.append({"feature": "risk_composite_score", "shap": 0.05, "text": TMPLS["risk_composite_score"]})
    candidates.append({"feature": "interact_volume_x_reimb", "shap": 0.04, "text": TMPLS["interact_volume_x_reimb"]})

    # Sort by shap descending and return top 5
    candidates.sort(key=lambda x: x["shap"], reverse=True)
    return candidates[:5]


def get_risk_tier(score):
    if   score >= 0.85:             return "CRITICAL", "#E84C4C", "Immediate investigation required — block claim"
    elif score >= THRESHOLD:        return "HIGH",     "#FF8C42", "Flag for detailed manual review before approval"
    elif score >= THRESHOLD * 0.7:  return "MEDIUM",   "#F5C842", "Process with additional verification steps"
    else:                           return "LOW",       "#22C97A", "Approve — claim is within normal parameters"


# ═════════════════════════════════════════════════════════════════
#  ROUTES
# ═════════════════════════════════════════════════════════════════

@app.route("/")
def index():
    return render_template(
        "index.html",
        model_auc=METADATA["val_auc_roc"],
        threshold=THRESHOLD,
        model_f1=METADATA["val_f1"],
    )


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data    = request.get_json(force=True, silent=True) or {}
        feat_df = build_feature_df(data)
        score, xgb_s, lgbm_s, rf_s = ensemble_predict(feat_df)
        tier, color, action = get_risk_tier(score)
        reasons = get_shap_reasons(feat_df, data)

        return jsonify({
            "success"        : True,
            "fraud_score"    : score,
            "fraud_predicted": score >= THRESHOLD,
            "risk_tier"      : tier,
            "tier_color"     : color,
            "action"         : action,
            "xgb_score"      : xgb_s,
            "lgbm_score"     : lgbm_s,
            "rf_score"       : rf_s,
            "threshold"      : THRESHOLD,
            "reasons"        : reasons,
            "predicted_at"   : datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        })

    except Exception as exc:
        app.logger.error(traceback.format_exc())
        return jsonify({"success": False, "error": str(exc)}), 500


@app.route("/scan", methods=["POST"])
def scan_document():
    """Document scanner — manual field entry, no OCR, no PyTorch."""
    try:
        if request.is_json:
            data = request.get_json(force=True, silent=True) or {}
        else:
            data = {
                "ip_avg_reimbursement"    : request.form.get("ip_avg_reimbursement",      0),
                "ip_avg_stay_days"        : request.form.get("ip_avg_stay_days",           0),
                "ip_claim_count"          : request.form.get("ip_claim_count",             0),
                "ip_avg_chronic_cond"     : request.form.get("ip_avg_chronic_cond",        0),
                "total_unique_patients"   : request.form.get("total_unique_patients",      0),
                "ip_unique_attending_phys": request.form.get("ip_unique_attending_phys",   0),
            }

        feat_df = build_feature_df(data)
        score, xgb_s, lgbm_s, rf_s = ensemble_predict(feat_df)
        tier, color, action = get_risk_tier(score)
        reasons = get_shap_reasons(feat_df, data)

        return jsonify({
            "success"         : True,
            "fraud_score"     : score,
            "fraud_predicted" : score >= THRESHOLD,
            "risk_tier"       : tier,
            "tier_color"      : color,
            "action"          : action,
            "xgb_score"       : xgb_s,
            "lgbm_score"      : lgbm_s,
            "rf_score"        : rf_s,
            "extracted_fields": {k: str(v) for k, v in data.items() if v},
            "ocr_regions"     : 0,
            "reasons"         : reasons,
            "predicted_at"    : datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        })

    except Exception as exc:
        app.logger.error(traceback.format_exc())
        return jsonify({"success": False, "error": str(exc)}), 500


@app.route("/health")
def health():
    return jsonify({
        "status"      : "running",
        "model_auc"   : METADATA["val_auc_roc"],
        "model_f1"    : METADATA["val_f1"],
        "threshold"   : THRESHOLD,
        "features"    : len(FEATURE_COLS),
        "models_ready": labels,
        "shap_ready"  : SHAP_OK,
        "ocr"         : "disabled",
        "timestamp"   : datetime.now().isoformat(),
    })


# ── Error handlers ────────────────────────────────────────────────────────────
@app.errorhandler(404)
def not_found(e):
    return jsonify({"success": False, "error": "Endpoint not found"}), 404

@app.errorhandler(405)
def method_not_allowed(e):
    return jsonify({"success": False, "error": "Method not allowed"}), 405

@app.errorhandler(413)
def file_too_large(e):
    return jsonify({"success": False, "error": "File too large (max 16 MB)"}), 413

@app.errorhandler(500)
def server_error(e):
    return jsonify({"success": False, "error": str(e)}), 500


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n  Browser:      http://localhost:5000")
    print("  Health check: http://localhost:5000/health\n")
    app.run(debug=True, host="0.0.0.0", port=5000)
