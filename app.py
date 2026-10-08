import io, pickle, warnings, json
from itertools import combinations
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

warnings.filterwarnings("ignore")

EPS = 1e-6

###
# PATHS
###

BASE       = Path(__file__).parent
MODEL_DIR  = BASE / "models"
STATIC_DIR = BASE / "static"

###
# FILES LOADING
###

PIPELINE   = joblib.load(MODEL_DIR / "pipeline_final.pkl")
THRESHOLD  = pickle.load(open(MODEL_DIR / "threshold.pkl", "rb"))
FEAT_NAMES = pickle.load(open(MODEL_DIR / "feature_names.pkl", "rb"))
MODEL_META = json.loads((MODEL_DIR / "model_meta.json").read_text())

IMPUTER = PIPELINE.named_steps["imputer"]
SCALER  = PIPELINE.named_steps["scaler"]
CLF     = PIPELINE.named_steps["clf"]
COEFS   = np.array(MODEL_META["coefs"])

###
# CONSTANTS
###

REQUIRED_COLS = [
    "Biofilm","Biofilm_PEG","Auxins","Auxins_PEG",
    "ACC","ACC_PEG","Trehalose","Trehalose_PEG",
    "Proline","Proline_PEG","Antioxidants","Antioxidants_PEG",
    "Nitrogen","Nitrogen_PEG","Phosphate",
    "Potassium","Potassium_PEG",
    "Glycine_Betaine","Silica",
    "Manganese","Calcium","Polyamines","Zinc",
]

PAIRED_TRAITS = [
    ("Biofilm_PEG",     "Biofilm"),
    ("Auxins_PEG",      "Auxins"),
    ("ACC_PEG",         "ACC"),
    ("Trehalose_PEG",   "Trehalose"),
    ("Proline_PEG",     "Proline"),
    ("Antioxidants_PEG","Antioxidants"),
    ("Nitrogen_PEG",    "Nitrogen"),
    ("Potassium_PEG",   "Potassium"),
]

TRIPLET_FEATURE_NAMES = [f for f in FEAT_NAMES if f.startswith("trip_")]

SYNCOM_K        = 5 # Amount of bacteria per Syncom
SYNCOM_N        = 3 # Amount of Syncoms generated
SYNCOM_MIN_DIFF = 2 # How many bacteria each Syncom has to variate from the other

SYNCOM_MAX_POOL = 30 # How many top bacteria are used to form the Syncoms

SHAP_BACKGROUND = np.zeros(len(FEAT_NAMES))

###
# FEATURE ENGINEERING
###

def parse_triplet_name(name: str):
    """
    trip_A_x_B_div_C  -> ("AxBdC", A, B, C)   value = A * B / C
    trip_A_div_B_x_C  -> ("AdBxC", A, B, C)   value = A / (B * C)
    """
    inner = name[len("trip_"):]
    if "_x_" in inner.split("_div_")[0]:
        ab, c = inner.split("_div_", 1)
        a, b = ab.split("_x_", 1)
        return "AxBdC", a, b, c
    a, bc = inner.split("_div_", 1)
    b, c = bc.split("_x_", 1)
    return "AdBxC", a, b, c


def engineer_features(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    engineer_features computes the 3 different types of
    features that are used by the model
    """
    raw = df_raw[REQUIRED_COLS].copy()
    features = raw.copy()

    # Stress-response ratios
    for peg, reg in PAIRED_TRAITS:
        features[f"StressRatio_{reg}"] = raw[peg] / (raw[reg] + EPS)

    # Pairwise ratios - All biochemical columns
    for i in range(len(REQUIRED_COLS)):
        for j in range(i + 1, len(REQUIRED_COLS)):
            a, b = REQUIRED_COLS[i], REQUIRED_COLS[j]
            features[f"ratio_{a}_div_{b}"] = raw[a] / (raw[b] + EPS)

    # Triplet features
    for name in TRIPLET_FEATURE_NAMES:
        form, a, b, c = parse_triplet_name(name)
        if a in raw.columns and b in raw.columns and c in raw.columns:
            if form == "AxBdC":
                features[name] = (raw[a] * raw[b]) / (raw[c] + EPS)
            else:
                features[name] = raw[a] / (raw[b] * raw[c] + EPS)

    # Align exactly to what the model expects; anything still missing -> 0
    features = features.reindex(columns=FEAT_NAMES, fill_value=0.0)
    return features.replace([np.inf, -np.inf], np.nan)


def run_pipeline(imputer: pd.DataFrame):
    imputer  = IMPUTER.transform(imputer)
    scaler   = SCALER.transform(imputer)
    proba  = CLF.predict_proba(scaler)[:, 1]
    labels = np.where(proba >= THRESHOLD, "Beneficial", "Non-Beneficial")
    return proba, labels, scaler


def compute_shap(scaler: np.ndarray, top_n: int = 8):
    """
    compute-shap calculates per-strain SHAP values,
    exact for this linear model, relative to
    SHAP_BACKGROUND (the training set's center)
    """
    phi = COEFS * (scaler - SHAP_BACKGROUND)
    result = []
    for i in range(scaler.shape[0]):
        idx = np.argsort(np.abs(phi[i]))[::-1][:top_n]
        result.append([
            {"feature": FEAT_NAMES[j], "shap": round(float(phi[i][j]), 5)}
            for j in idx
        ])
    return result


def feature_category(feature_name: str) -> str:
    """
    feature_category divides features into their type, from
    the 3 previously generated and matches the frontend's
    theme-color keys in index.html's renderOverview()
    """
    if feature_name.startswith("trip_"):
        return "Triplet interaction"
    if feature_name.startswith("ratio_"):
        return "Pairwise ratio"
    if feature_name.startswith("StressRatio_"):
        return "Stress-response ratio"
    return "Raw biochemical trait"


def readable_label(feature_name: str) -> str:
    """
    readable_label strips the mechanical prefix and swaps
    underscores for spaces (display only)
    """
    label = feature_name
    for prefix in ("trip_", "ratio_", "StressRatio_"):
        if label.startswith(prefix):
            label = label[len(prefix):]
            break
    return label.replace("_", " ")


def compute_batch_importance(scaler: np.ndarray, top_n: int = 10):
    """
    compute_batch_importance computes the importance
    of each feature across all strains of the uploaded dataset,
    using mean absolute SHAP value (relative to SHAP_BACKGROUND) for each feature.
    """
    shap_values      = COEFS * (scaler - SHAP_BACKGROUND)
    absolute_shap_values  = np.abs(shap_values)
    mean_abs = absolute_shap_values.mean(axis=0)
    std_abs  = absolute_shap_values.std(axis=0)
    idx = np.argsort(-mean_abs)[:top_n]
    return [
        {
            "label":      readable_label(FEAT_NAMES[i]),
            "theme":      feature_category(FEAT_NAMES[i]),
            "importance": float(mean_abs[i]),
            "std":        float(std_abs[i]),
        }
        for i in idx
    ]

###
# SYNCOM ASSEMBLY
###

def syncom_assembly(ranking_df: pd.DataFrame) -> list:
    """
    syncom_assembly takes the top 30 predicted-beneficial strains,
    by P_Beneficial searches all 5-strain combinations to select the top 3
    syncoms, by mean P, and also assuring that between syncoms, there is
    at least a difference of 2 strains
    """

    pool = ranking_df[ranking_df["P_Beneficial"] >= THRESHOLD].copy()
    pool = pool.sort_values("P_Beneficial", ascending=False).head(SYNCOM_MAX_POOL)
    ids  = pool["Bacteria_ID"].tolist()

    if len(ids) < SYNCOM_K:
        return []

    p_map  = dict(zip(pool["Bacteria_ID"], pool["P_Beneficial"]))
    scored = sorted(
        ((combo, float(np.mean([p_map[s] for s in combo])))
         for combo in combinations(ids, SYNCOM_K)),
        key=lambda x: -x[1]
    )

    selected = []
    for combo, score in scored:
        if len(selected) >= SYNCOM_N:
            break
        cs = set(combo)
        if all(len(cs.symmetric_difference(set(prev))) >= SYNCOM_MIN_DIFF * 2
               for prev, _ in selected):
            selected.append((combo, score))

    return [
        {
            "name":    f"Elite_SC{i+1}",
            "strains": list(combo),
            "mean_p":  round(score, 4),
            "min_p":   round(min(p_map[s] for s in combo), 4),
            "max_p":   round(max(p_map[s] for s in combo), 4),
        }
        for i, (combo, score) in enumerate(selected)
    ]


###
# FAST API
###

app = FastAPI(title="SAMBA", version="1.0")

@app.get("/api/model-info")
def model_info():
    return {
        "model":        "ElasticNet (LogisticRegression, saga)",
        "threshold":    float(THRESHOLD),
        "trained_on":   "124 maize rhizosphere strains",
        "required_cols": REQUIRED_COLS,
    }


@app.post("/api/analyze")
async def analyze(file: UploadFile = File(...)):
    try:
        content = await file.read()
        df_in   = pd.read_csv(io.BytesIO(content))
    except Exception as e:
        raise HTTPException(400, f"Could not parse CSV: {e}")

    missing = [c for c in REQUIRED_COLS if c not in df_in.columns]
    if missing:
        return JSONResponse({
            "status":  "error",
            "message": f"Missing {len(missing)} required column(s): {', '.join(missing[:5])}{'…' if len(missing) > 5 else ''}",
            "missing": missing,
        })

    if "Bacteria_ID" not in df_in.columns:
        df_in.insert(0, "Bacteria_ID", [f"Strain_{i+1}" for i in range(len(df_in))])

    X_eng               = engineer_features(df_in)
    proba, labels, scaler = run_pipeline(X_eng)
    shap_list           = compute_shap(scaler, top_n=8)
    batch_importance     = compute_batch_importance(scaler, top_n=10)

    ranking = (
        pd.DataFrame({
            "Bacteria_ID":  df_in["Bacteria_ID"].astype(str).values,
            "P_Beneficial": np.round(proba, 6),
            "Predicted":    labels,
        })
        .sort_values("P_Beneficial", ascending=False)
        .reset_index(drop=True)
    )
    ranking["Rank"] = ranking.index + 1

    id_to_shap = {
        str(df_in["Bacteria_ID"].iloc[i]): shap_list[i]
        for i in range(len(df_in))
    }

    trait_rows = {
        str(row["Bacteria_ID"]): {c: round(float(row[c]), 4) for c in REQUIRED_COLS}
        for _, row in df_in.iterrows()
    }

    syncoms      = syncom_assembly(ranking)
    n_total      = len(ranking)
    n_beneficial = int((ranking["Predicted"] == "Beneficial").sum())

    return JSONResponse({
        "status":          "ok",
        "n_total":         n_total,
        "n_beneficial":    n_beneficial,
        "n_non":           n_total - n_beneficial,
        "ranking":         ranking.to_dict(orient="records"),
        "syncoms":         syncoms,
        "trait_rows":      trait_rows,
        "shap_top":        id_to_shap,
        "feat_importance": batch_importance,
    })


###
# FRONTEND
###

app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")


###
# ENTRY POINT
###

if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info",
    )