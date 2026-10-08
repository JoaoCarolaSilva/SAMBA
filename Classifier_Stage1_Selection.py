import warnings, os, joblib
import numpy as np
import pandas as pd
from copy import deepcopy
from datetime import datetime
from sklearn.model_selection import train_test_split, LeaveOneOut, cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.preprocessing import RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score, roc_auc_score, recall_score, precision_score
from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE
from imblearn.combine import SMOTETomek, SMOTEENN
from imblearn.pipeline import Pipeline as ImbPipeline

warnings.filterwarnings("ignore")
try:
    import ctypes; ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
except Exception:
    pass

output_dir = f"Classifier_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
os.makedirs(output_dir, exist_ok=True)
print(f"[START] Output -> {output_dir}/")

###
# DATA & BIOLOGICAL TARGET
###

df = pd.read_csv("Clean_data.csv")

pheno_cols_for_target = [column for column in df.columns if "Shoot" in column or "Root" in column]
df[pheno_cols_for_target] = df[pheno_cols_for_target].fillna(
    df[pheno_cols_for_target].mean())

CONTROL_RAW = {
    "Shoot_Length_Drought": 57.5,
    "Root_Length_Drought":  60.9,
    "Shoot_Weight_Drought": 13.0,
    "Root_Weight_Drought":  10.8,
}
BENEFIT_THRESHOLD = 0.10
W_LENGTH          = 0.40
W_WEIGHT          = 0.60

drought_cols        = list(CONTROL_RAW.keys())
drought_length_cols = [column for column in CONTROL_RAW if "Length" in column]
drought_weight_cols = [column for column in CONTROL_RAW if "Weight" in column]

def classify_beneficial(row):
    """
    classify_beneficial is responsible for defining the
    threshold for which a bacteria is considered beneficial
    """
    orig      = {col: row[col] + CONTROL_RAW[col] for col in drought_cols}
    len_votes = sum(orig[c] > CONTROL_RAW[c] * (1 + BENEFIT_THRESHOLD)
                    for c in drought_length_cols)
    weight_votes = sum(orig[c] > CONTROL_RAW[c] * (1 + BENEFIT_THRESHOLD)
                    for c in drought_weight_cols)
    beneficial_score = len_votes * W_LENGTH + weight_votes * W_WEIGHT
    return int(beneficial_score >= 0.5)

df["Target"] = df[drought_cols].apply(classify_beneficial, axis=1)
n_pos   = df["Target"].sum()
n_neg   = (df["Target"] == 0).sum()
y       = df["Target"].copy()

print(f"  [TARGET] Voting: strain beneficial if weighted majority of traits "
      f"exceed control by >{int(BENEFIT_THRESHOLD*100)}% (L:{W_LENGTH:.0%} / W:{W_WEIGHT:.0%})")
print(f"  [DATA] {len(df)} samples | {n_pos} Beneficial / {n_neg} Non-Beneficial "
      f"| ratio {n_neg/n_pos:.2f}")

###
# FEATURE ENGINEERING
###

phenotype_cols = [column for column in df.columns if "Shoot" in column or "Root" in column]
biochem_cols   = [column for column in df.columns
                  if column not in ("Bacteria_ID", "Target")
                  and column not in phenotype_cols]
X = df[biochem_cols].copy()

for peg, reg in [("Biofilm_PEG",     "Biofilm"),
                 ("Auxins_PEG",      "Auxins"),
                 ("ACC_PEG",         "ACC"),
                 ("Trehalose_PEG",   "Trehalose"),
                 ("Proline_PEG",     "Proline"),
                 ("Antioxidants_PEG","Antioxidants"),
                 ("Nitrogen_PEG",    "Nitrogen"),
                 ("Potassium_PEG",   "Potassium")]:
    if peg in df.columns:
        X[f"StressRatio_{reg}"] = df[peg] / (df[reg] + 1e-6)


for i in range(len(biochem_cols)):
    for j in range(i + 1, len(biochem_cols)):
        X[f"ratio_{biochem_cols[i]}_div_{biochem_cols[j]}"] = (
            df[biochem_cols[i]] / (df[biochem_cols[j]] + 1e-6))

###
# TRIPLET FEATURES
###


TRIPLET_SCREENING_CSV = "Triplet_results_20260917_231854/triplet_cv_screening.csv"
TRIPLET_N = 17

def add_triplets(features_df, raw_df, screening_csv, n_top):
    """
    add_triplets adds the triplet features computed in the
    Additional_Feature_Testing.py script
    """

    import os
    if not os.path.exists(screening_csv):
        print(f"  [TRIPLETS] File not found: {screening_csv} — skipping")
        return features_df, 0
    trip_df = pd.read_csv(screening_csv)
    positives = trip_df[trip_df["Delta_F1"] > 0].head(n_top)
    added = 0
    for _, row in positives.iterrows():
        name = row["Feature"]

        inner = name[len("trip_"):]
        if "_x_" in inner.split("_div_")[0]:
            ab, c = inner.split("_div_", 1)
            a, b  = ab.split("_x_", 1)
            if a in raw_df.columns and b in raw_df.columns and c in raw_df.columns:
                features_df[name] = (raw_df[a] * raw_df[b]) / (raw_df[c] + 1e-6)
                added += 1
        else:
            a, bc = inner.split("_div_", 1)
            b, c  = bc.split("_x_", 1)
            if a in raw_df.columns and b in raw_df.columns and c in raw_df.columns:
                features_df[name] = raw_df[a] / (raw_df[b] * raw_df[c] + 1e-6)
                added += 1
    return features_df, added

X, n_added = add_triplets(X, df, TRIPLET_SCREENING_CSV, TRIPLET_N)
print(f"  [TRIPLETS] top{TRIPLET_N} config: {n_added} features added")

X             = X.replace([np.inf, -np.inf], np.nan)
feature_names = X.columns.tolist()
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

print(f"  [FEATURES] {len(feature_names)} total features")

###
# MODEL COMPARISON
###

loo = LeaveOneOut()

def make_model_pipe(model):
    """
    make_model_pipe is used to build a fixed preprocessing pipeline
    around any classifier passed in
    """
    return ImbPipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  RobustScaler()),
        ("clf",     deepcopy(model)),
    ])

candidate_models = {
    "LogisticRegression_ElasticNet": LogisticRegression(
        class_weight="balanced", C=0.5, max_iter=2000, random_state=42,
        solver="saga", penalty="elasticnet", l1_ratio=0.5),
    "RandomForest": RandomForestClassifier(
        n_estimators=300, class_weight="balanced", random_state=42),
    "XGBoost": XGBClassifier(
        n_estimators=300, eval_metric="logloss", random_state=42,
        scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum()),
    "SVM_RBF": SVC(
        kernel="rbf", class_weight="balanced", probability=True, random_state=42),
}

model_comparison_results = []
print("\n  [0] Model comparison (training set only):")
for name, model in candidate_models.items():
    try:
        p     = make_model_pipe(model)
        preds = cross_val_predict(p, X_train, y_train, cv=loo)
        proba_method = "decision_function" if name == "SVM_RBF" else "predict_proba"
        proba = cross_val_predict(deepcopy(p), X_train, y_train, cv=loo,
                                   method=proba_method)
        if proba.ndim > 1:
            proba = proba[:, 1]

        f1  = f1_score(y_train, preds, average="macro")
        auc = roc_auc_score(y_train, proba)
        r1  = recall_score(y_train, preds, pos_label=1)
        p1  = precision_score(y_train, preds, pos_label=1, zero_division=0)
        model_comparison_results.append({"Model": name, "LOO_F1": f1, "LOO_AUC": auc,
                                          "Recall_1": r1, "Precision_1": p1})
        print(f"       {name:30s} | F1={f1:.4f}  AUC={auc:.4f}  R1={r1:.4f}")
    except Exception as e:
        print(f"       {name:30s} | FAILED: {e}")

model_comp_df = pd.DataFrame(model_comparison_results).sort_values("LOO_F1", ascending=False)
model_comp_df.to_csv(f"{output_dir}/model_comparison.csv", index=False)
print(f"  [0] Winner: {model_comp_df.iloc[0]['Model']} "
      f"(LOO F1={model_comp_df.iloc[0]['LOO_F1']:.4f})\n")

###
# AUGMENTATION & FEATURE-COUNT GRID
###

def make_pipe(sampler=None):
    """
    make_pipe builds a complete imbalanced-classification pipeline with
    a fixed logistic regression classifier and an optional resampling step
    """
    steps = [("imputer", SimpleImputer(strategy="median")),
             ("scaler",  RobustScaler())]
    if sampler is not None:
        steps.append(("sampler", deepcopy(sampler)))
    steps.append(("clf", LogisticRegression(
        class_weight="balanced", C=0.5, max_iter=2000,
        random_state=42, solver="saga", penalty="elasticnet", l1_ratio=0.5)))
    return ImbPipeline(steps)

samplers = {
    "None":           None,
    "SMOTE_k3":       SMOTE(k_neighbors=3, random_state=42),
    "SMOTE_k4":       SMOTE(k_neighbors=4, random_state=42),
    "BorderlineSMOTE":BorderlineSMOTE(random_state=42),
    "ADASYN":         ADASYN(random_state=42),
    "SMOTETomek":     SMOTETomek(random_state=42),
    "SMOTEENN":       SMOTEENN(random_state=42),
}

from sklearn.feature_selection import SelectKBest, f_classif

grid_results = []
best_sampler_name, best_sampler = "None", None
best_n_feat = len(feature_names)
best_grid_score = (-1.0, -1.0, -1.0)

print("\n  [1] Augmentation & feature-count grid:")
for name, sampler in samplers.items():
    for n in sorted([10, 20, 30, 50, 100, 200, len(feature_names)]):
        try:
            steps_sel = [("imputer",  SimpleImputer(strategy="median")),
                         ("selector", SelectKBest(f_classif, k=min(n, len(feature_names)))),
                         ("scaler",   RobustScaler())]
            if sampler is not None:
                steps_sel.append(("sampler", deepcopy(sampler)))
            steps_sel.append(("clf", LogisticRegression(
                class_weight="balanced", C=0.5, max_iter=2000,
                random_state=42, solver="saga", penalty="elasticnet", l1_ratio=0.5)))
            pipe_sel = ImbPipeline(steps_sel)
            preds    = cross_val_predict(pipe_sel, X, y, cv=loo)
            proba    = cross_val_predict(deepcopy(pipe_sel), X, y, cv=loo, method="predict_proba")[:, 1]
            f1       = f1_score(y, preds, average="macro")
            auc      = roc_auc_score(y, proba)
            r1       = recall_score(y, preds, pos_label=1)
            p1       = precision_score(y, preds, pos_label=1, zero_division=0)
            grid_results.append({"Strategy": name, "N_Features": n, "LOO_F1": f1,
                                  "LOO_AUC": auc, "Recall_1": r1, "Precision_1": p1})
            candidate_score = (f1, r1, auc)
            if candidate_score > best_grid_score:
                best_grid_score = candidate_score
                best_sampler_name, best_sampler, best_n_feat = name, sampler, n
            print(f"       {name:20s} N={n:4d} | F1={f1:.4f}  AUC={auc:.4f}  R1={r1:.4f}")
        except Exception as e:
            print(f"       {name:20s} N={n:4d} | FAILED: {e}")

grid_df = pd.DataFrame(grid_results).sort_values("LOO_F1", ascending=False)
grid_df.to_csv(f"{output_dir}/augmentation_feature_grid.csv", index=False)
print(f"  [1] Winner: {best_sampler_name} + N={best_n_feat} "
      f"(F1={best_grid_score[0]:.4f}  R1={best_grid_score[1]:.4f}  AUC={best_grid_score[2]:.4f})")

if best_n_feat < len(feature_names):

    selector_final = SelectKBest(f_classif, k=best_n_feat).fit(X_train, y_train)
    X_final   = pd.DataFrame(selector_final.transform(X))
    X_tr_fin  = pd.DataFrame(selector_final.transform(X_train))
    X_te_fin  = pd.DataFrame(selector_final.transform(X_test))
    feat_names_final = [feature_names[i] for i in selector_final.get_support(indices=True)]
else:
    X_final, X_tr_fin, X_te_fin = X, X_train, X_test
    feat_names_final = feature_names

###
# HANDOFF TO STAGE 2
###

stage1_state = {
    "df":                df,
    "X_final":           X_final,
    "X_tr_fin":          X_tr_fin,
    "X_te_fin":          X_te_fin,
    "y":                 y,
    "y_train":           y_train,
    "y_test":            y_test,
    "feat_names_final":  feat_names_final,
    "best_sampler":      best_sampler,
    "best_sampler_name": best_sampler_name,
    "best_n_feat":       best_n_feat,
    "grid_df":           grid_df,
    "n_pos":             n_pos,
    "n_neg":             n_neg,
    "CONTROL_RAW":       CONTROL_RAW,
    "BENEFIT_THRESHOLD": BENEFIT_THRESHOLD,
    "W_LENGTH":          W_LENGTH,
    "W_WEIGHT":          W_WEIGHT,
}
joblib.dump(stage1_state, f"{output_dir}/stage1_state.pkl")
print(f"\n[END] Stage 1 done -> {output_dir}/stage1_state.pkl")
print(f"       Set STAGE1_DIR = \"{output_dir}\" at the top of Classifier_Stage2_Optuna_SHAP.py")
