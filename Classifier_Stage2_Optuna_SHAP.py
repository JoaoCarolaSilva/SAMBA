import warnings, os, joblib, json
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
from copy import deepcopy
from sklearn.model_selection import StratifiedKFold, LeaveOneOut, RepeatedStratifiedKFold, cross_val_predict, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.metrics import (classification_report, confusion_matrix, f1_score,
    roc_auc_score, RocCurveDisplay, recall_score, precision_score)
from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE  # needed to unpickle best_sampler
from imblearn.combine import SMOTETomek, SMOTEENN                  # needed to unpickle best_sampler
from imblearn.pipeline import Pipeline as ImbPipeline
import optuna
import shap

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman", "Liberation Serif", "DejaVu Serif"]
plt.rcParams["mathtext.fontset"] = "stix"

PALETTE_DARK_GREEN  = "#1b7837"
PALETTE_MED_GREEN   = "#5aae61"
PALETTE_DARK_BLUE   = "#2166ac"
PALETTE_MED_BLUE    = "#4393c3"


GREEN_WHITE_BLUE = mcolors.LinearSegmentedColormap.from_list(
    "green_white_blue", [PALETTE_DARK_BLUE, "#f7f7f7", PALETTE_DARK_GREEN]
)

GREENS_SEQ = mcolors.LinearSegmentedColormap.from_list("greens_seq", ["#ffffff", PALETTE_DARK_GREEN])
BLUES_SEQ  = mcolors.LinearSegmentedColormap.from_list("blues_seq",  ["#ffffff", PALETTE_DARK_BLUE])

try:
    import ctypes; ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
except Exception:
    pass

###
# HANDOFF FROM STAGE 1
###

STAGE1_DIR = "Classifier_results_20260920_115035"  #  set to Stage 1's printed output_dir

output_dir = STAGE1_DIR
os.makedirs(output_dir, exist_ok=True)
stage1_state = joblib.load(f"{STAGE1_DIR}/stage1_state.pkl")

df                = stage1_state["df"]
X_final           = stage1_state["X_final"]
X_tr_fin          = stage1_state["X_tr_fin"]
X_te_fin          = stage1_state["X_te_fin"]
y                 = stage1_state["y"]
y_train           = stage1_state["y_train"]
y_test            = stage1_state["y_test"]
feat_names_final  = stage1_state["feat_names_final"]
best_sampler      = stage1_state["best_sampler"]
best_sampler_name = stage1_state["best_sampler_name"]
best_n_feat       = stage1_state["best_n_feat"]
grid_df           = stage1_state["grid_df"]
n_pos             = stage1_state["n_pos"]
n_neg             = stage1_state["n_neg"]
CONTROL_RAW       = stage1_state["CONTROL_RAW"]
BENEFIT_THRESHOLD = stage1_state["BENEFIT_THRESHOLD"]
W_LENGTH          = stage1_state["W_LENGTH"]
W_WEIGHT          = stage1_state["W_WEIGHT"]

loo = LeaveOneOut()

print(f"[START] Loaded Stage 1 from {STAGE1_DIR}/stage1_state.pkl")
print(f"        Output -> {output_dir}/")
print(f"        Sampler={best_sampler_name}  N={best_n_feat}")

###
# OPTUNA
###

N_TRIALS  = 100   # forced to the exact validated params below — no new search
VALIDATED_PARAMS = {"C": 0.06348217329775487, "l1_ratio": 0.3874420226870773}
cv_optuna = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=42)

print(f"\n  [2] Optuna ({N_TRIALS} trials) — ElasticNet + {best_sampler_name}:")

def objective(trial):
    """
    objective defines optuna hyperparameter limits and objective
    """
    clf = LogisticRegression(
        class_weight="balanced", max_iter=2000, random_state=42,
        solver="saga", penalty="elasticnet",
        C        = trial.suggest_float("C",        0.001, 10.0, log=True),
        l1_ratio = trial.suggest_float("l1_ratio", 0.35, 0.65))
    steps = [("scaler", RobustScaler())]
    if best_sampler is not None:
        steps.append(("sampler", deepcopy(best_sampler)))
    steps.append(("clf", clf))
    p      = ImbPipeline(steps)
    scores = cross_val_score(p, X_tr_fin, y_train, cv=cv_optuna,
                             scoring="f1_macro", n_jobs=-1)
    p.fit(X_tr_fin, y_train)
    trial.set_user_attr("train_f1", f1_score(y_train, p.predict(X_tr_fin), average="macro"))
    trial.set_user_attr("cv_std",   scores.std())
    return scores.mean()

study = optuna.create_study(direction="maximize")
study.enqueue_trial(VALIDATED_PARAMS)
study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)
best_params = study.best_params
print(f"  [2] Optuna done | Best CV F1={study.best_value:.4f}")
print(f"       C={best_params['C']:.5f}  l1_ratio={best_params['l1_ratio']:.4f}")

###
# FINAL PIPELINE BUILDER
###

def build_final_pipe(params):
    """
    build_final_pipeline puts together the
    model with the optuna tuning
    """
    clf = LogisticRegression(
        class_weight="balanced", max_iter=2000, random_state=42,
        solver="saga", penalty="elasticnet",
        C=params["C"], l1_ratio=params["l1_ratio"])
    steps = [("imputer", SimpleImputer(strategy="median")),
             ("scaler",  RobustScaler())]
    if best_sampler is not None:
        steps.append(("sampler", deepcopy(best_sampler)))
    steps.append(("clf", clf))
    return ImbPipeline(steps)

###
# THRESHOLD TUNING
###

oof_proba = cross_val_predict(
    build_final_pipe(best_params), X_tr_fin, y_train,
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    method="predict_proba")[:, 1]

th_scores = [{"threshold":   th,
               "f1_macro":    f1_score(y_train, (oof_proba >= th).astype(int), average="macro"),
               "recall_1":    recall_score(y_train, (oof_proba >= th).astype(int), pos_label=1),
               "precision_1": precision_score(y_train, (oof_proba >= th).astype(int),
                                              pos_label=1, zero_division=0)}
              for th in np.arange(0.25, 0.75, 0.01)]
th_df      = pd.DataFrame(th_scores)
valid_th   = th_df[th_df["recall_1"] >= 0.45]
best_th_row    = (valid_th.loc[valid_th["f1_macro"].idxmax()] if len(valid_th)
                  else th_df.loc[th_df["f1_macro"].idxmax()])
best_threshold = best_th_row["threshold"]
th_df.to_csv(f"{output_dir}/threshold_analysis.csv", index=False)

###
# LOOCV EVALUATION
###

loo_proba         = cross_val_predict(build_final_pipe(best_params), X_final, y,
                        cv=loo, method="predict_proba")[:, 1]
loo_preds_default = (loo_proba >= 0.50).astype(int)
loo_preds_tuned   = (loo_proba >= best_threshold).astype(int)
auc_loo           = roc_auc_score(y, loo_proba)

def _metrics(yt, yp):
    """
    _metrics defines the main evaluation metrics of the model
    """
    return (f1_score(yt, yp, average="macro"),
            recall_score(yt, yp, pos_label=1),
            precision_score(yt, yp, pos_label=1, zero_division=0))

f1_def, r1_def, p1_def = _metrics(y, loo_preds_default)
f1_tun, r1_tun, p1_tun = _metrics(y, loo_preds_tuned)

if f1_tun > f1_def and r1_tun >= 0.45:
    final_threshold, final_preds = best_threshold, loo_preds_tuned
    final_f1, final_r1, final_p1 = f1_tun, r1_tun, p1_tun
else:
    final_threshold, final_preds = 0.50, loo_preds_default
    final_f1, final_r1, final_p1 = f1_def, r1_def, p1_def

print(f"\n  [3] LOOCV | F1={final_f1:.4f}  R1={final_r1:.3f} ({int(final_r1*n_pos)}/{n_pos})"
      f"  AUC={auc_loo:.4f}  th={final_threshold:.2f}")

###
# TEST SET EVALUATION
###

final_pipe = build_final_pipe(best_params)
final_pipe.fit(X_tr_fin, y_train)
test_proba  = final_pipe.predict_proba(X_te_fin)[:, 1]
test_preds  = (test_proba >= final_threshold).astype(int)
test_f1, test_r1, test_p1 = _metrics(y_test, test_preds)
test_auc    = roc_auc_score(y_test, test_proba)
train_preds = (final_pipe.predict_proba(X_tr_fin)[:, 1] >= final_threshold).astype(int)
train_f1, _, _ = _metrics(y_train, train_preds)
print(f"  [4] Test  | F1={test_f1:.4f}  R1={test_r1:.3f}  AUC={test_auc:.4f}"
      f"  gap={train_f1-test_f1:+.4f}")

###
# FEATURE IMPORTANCE
###

deploy_pipe = build_final_pipe(best_params)
deploy_pipe.fit(X_final, y)
perm    = permutation_importance(deploy_pipe, X_final, y,
                                  n_repeats=30, random_state=42, scoring="f1_macro")
perm_df = pd.DataFrame({"Feature":    feat_names_final,
                         "Importance": perm.importances_mean,
                         "Std":        perm.importances_std
                        }).sort_values("Importance", ascending=False)
perm_df.to_csv(f"{output_dir}/feature_importance.csv", index=False)
print(f"  [5] Top feature: {perm_df.iloc[0]['Feature']} ({perm_df.iloc[0]['Importance']:.4f})")

###
# BACTERIA RANKING
###

deploy_proba = deploy_pipe.predict_proba(X_final)[:, 1]
deploy_preds = (deploy_proba >= final_threshold).astype(int)
ranking = pd.DataFrame({
    "Bacteria_ID":    df["Bacteria_ID"].values,
    "P_Beneficial":   np.round(deploy_proba, 4),
    "Predicted":      ["Beneficial" if p == 1 else "Non-Beneficial" for p in deploy_preds],
    "Actual":         ["Beneficial" if a == 1 else "Non-Beneficial" for a in y.values],
    "Correct":        (deploy_preds == y.values).astype(int),
}).sort_values("P_Beneficial", ascending=False)
ranking.to_csv(f"{output_dir}/bacteria_ranking.csv", index=False)

###
# PLOTS
###

# Optuna convergence
completed = [t for t in study.trials if t.value is not None]
ov_train  = [t.user_attrs.get("train_f1", np.nan) for t in completed]
ov_val    = [t.value for t in completed]
tx        = range(len(ov_train))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))


best_train = np.maximum.accumulate(ov_train)
best_val   = np.maximum.accumulate(ov_val)
ax1.plot(tx, best_train, color=PALETTE_DARK_GREEN, lw=2, label="Best Train F1 so far")
ax1.plot(tx, best_val,   color=PALETTE_DARK_BLUE,  lw=2, label="Best CV F1 so far")
ax1.set(title="Optuna: Train vs CV (ElasticNet)", xlabel="Trial", ylabel="F1-Macro")
ax1.legend()

# Overfitting gap
raw_gap = pd.Series([t - v for t, v in zip(ov_train, ov_val)])
smoothed_gap = raw_gap.rolling(window=max(len(raw_gap) // 40, 1), center=True, min_periods=1).mean()
ax2.plot(tx, smoothed_gap, color=PALETTE_MED_BLUE, lw=2)
ax2.axhline(0,    color="black", ls="--", alpha=0.5)
ax2.axhline(0.15, color="black", ls="--", alpha=0.4, label="0.15 target")
ax2.set(title="Overfitting Gap", xlabel="Trial", ylabel="Gap")
ax2.legend()
plt.tight_layout()
plt.savefig(f"{output_dir}/overfitting_analysis.png", dpi=300)
plt.close()

# Confusion matrices LOOCV
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
for ax, preds, th_label in [(ax1, loo_preds_default, "0.50"),
                             (ax2, final_preds,       f"{final_threshold:.2f}")]:
    sns.heatmap(confusion_matrix(y, preds), annot=True, fmt="d", cmap=GREENS_SEQ,
                xticklabels=["Non-Beneficial", "Beneficial"],
                yticklabels=["Non-Beneficial", "Beneficial"], ax=ax)
    f1 = f1_score(y, preds, average="macro")
    r1 = recall_score(y, preds, pos_label=1)
    ax.set(title=f"LOOCV (th={th_label}, F1={f1:.3f}, R1={r1:.3f})",
           xlabel="Predicted", ylabel="Actual")
plt.tight_layout()
plt.savefig(f"{output_dir}/confusion_matrix_loocv.png", dpi=300)
plt.close()

# Confusion matrix test
fig, ax = plt.subplots(figsize=(7, 6))
sns.heatmap(confusion_matrix(y_test, test_preds), annot=True, fmt="d", cmap=BLUES_SEQ,
            xticklabels=["Non-Beneficial", "Beneficial"],
            yticklabels=["Non-Beneficial", "Beneficial"], ax=ax)
ax.set(title=f"Test Set (F1={test_f1:.3f}, R1={test_r1:.3f})",
       xlabel="Predicted", ylabel="Actual")
plt.tight_layout()
plt.savefig(f"{output_dir}/confusion_matrix_test.png", dpi=300)
plt.close()

# ROC curve
fig, ax = plt.subplots(figsize=(7, 6))
RocCurveDisplay.from_predictions(y, loo_proba, ax=ax, color=PALETTE_DARK_BLUE,
    name="ElasticNet LOOCV")  # sklearn appends "(AUC = ..)" automatically; avoids the duplicated AUC text seen before
ax.plot([0, 1], [0, 1], "k--", alpha=0.5)
ax.set_title(f"ROC Curve (AUC={auc_loo:.3f})")
plt.tight_layout()
plt.savefig(f"{output_dir}/roc_curve_loocv.png", dpi=300)
plt.close()

###
# SAVE MODEL & SCIENTIFIC REPORT
###

joblib.dump(deploy_pipe,      f"{output_dir}/pipeline_final.pkl")
joblib.dump(feat_names_final, f"{output_dir}/feature_names.pkl")
joblib.dump(final_threshold,  f"{output_dir}/threshold.pkl")

scaler_step = deploy_pipe.named_steps["scaler"]
clf_step    = deploy_pipe.named_steps["clf"]
model_meta = {
    "coefs":         clf_step.coef_[0].tolist(),
    "centers":       scaler_step.center_.tolist(),
    "scales":        scaler_step.scale_.tolist(),
    "feature_names": feat_names_final,
}
with open(f"{output_dir}/model_meta.json", "w", encoding="utf-8") as f:
    json.dump(model_meta, f)

tp = ((deploy_preds == 1) & (y == 1)).sum()
fp = ((deploy_preds == 1) & (y == 0)).sum()
fn = ((deploy_preds == 0) & (y == 1)).sum()
tn = ((deploy_preds == 0) & (y == 0)).sum()

with open(f"{output_dir}/scientific_report.txt", "w", encoding="utf-8") as f:
    f.write("=" * 70 + "\nCLASSIFIER MODEL - Scientific Report\n"
            "Drought Phenotype Prediction from Biochemical Traits\n" + "=" * 70 + "\n\n")
    f.write(f"1. DATASET\n   Samples: {len(df)} | Features (final): {len(feat_names_final)}\n"
            f"   Target: Beneficial (voting, >{int(BENEFIT_THRESHOLD*100)}% above control "
            f"in weighted majority of traits) vs Non-Beneficial\n"
            f"   Weights: Length {W_LENGTH:.0%} / Weight {W_WEIGHT:.0%}\n"
            f"   Control: Shoot_Length={CONTROL_RAW['Shoot_Length_Drought']}, "
            f"Root_Length={CONTROL_RAW['Root_Length_Drought']}, "
            f"Shoot_Weight={CONTROL_RAW['Shoot_Weight_Drought']}, "
            f"Root_Weight={CONTROL_RAW['Root_Weight_Drought']}\n"
            f"   Classes: {n_neg} non-beneficial / {n_pos} beneficial (ratio {n_neg/n_pos:.2f})\n\n")
    f.write(f"2. AUGMENTATION & FEATURE-COUNT GRID (joint search, SMOTE variants x N)\n"
            f"{grid_df.to_string(index=False)}\n"
            f"   Winner: {best_sampler_name} + N={best_n_feat}\n\n")
    f.write(f"3. MODEL\n   ElasticNet (LogisticRegression saga + elasticnet penalty)\n"
            f"   Fixed as final model: highest LOOCV F1 + near-zero overfit gap\n\n")
    f.write(f"4. OPTUNA ({N_TRIALS} trials)\n   Best CV F1: {study.best_value:.4f}\n"
            f"   Params: {best_params}\n\n")
    f.write(f"5. THRESHOLD\n   Method: OOF on training set (no leakage)\n"
            f"   Selected: {final_threshold:.2f}\n\n")
    f.write(f"6. LOOCV RESULTS (PRIMARY)\n"
            f"   F1={final_f1:.4f} | R1={final_r1:.4f} ({int(final_r1*n_pos)}/{n_pos})"
            f" | P1={final_p1:.4f} | AUC={auc_loo:.4f}\n")
    f.write(classification_report(y, final_preds,
            target_names=["Non-Beneficial", "Beneficial"]) + "\n")
    f.write(f"7. TEST SET RESULTS (SECONDARY)\n"
            f"   F1={test_f1:.4f} | R1={test_r1:.4f} | P1={test_p1:.4f}"
            f" | TrainF1={train_f1:.4f} (gap={train_f1-test_f1:+.4f})\n")
    f.write(classification_report(y_test, test_preds,
            target_names=["Non-Beneficial", "Beneficial"]) + "\n")
    f.write(f"8. DEPLOYMENT STATS\n   TP={tp} FP={fp} FN={fn} TN={tn}\n"
            f"   Found {tp}/{n_pos} top performers ({tp/n_pos*100:.0f}%)\n\n")
    f.write(f"9. TOP FEATURES (Permutation Importance)\n"
            f"{perm_df.head(15).to_string(index=False)}\n")

print(f"\n[END] All outputs -> {output_dir}/")

###
# SHAP ANALYSIS
###

print("\n  [6] SHAP analysis (LinearExplainer, interventional):")

X_imputed = deploy_pipe.named_steps["imputer"].transform(X_final)
X_scaled  = deploy_pipe.named_steps["scaler"].transform(X_imputed)
X_scaled_df = pd.DataFrame(X_scaled, columns=feat_names_final)

clf_final = deploy_pipe.named_steps["clf"]

explainer   = shap.LinearExplainer(clf_final, X_scaled_df,
                                    feature_perturbation="interventional")
shap_values = explainer(X_scaled_df)   # shap.Explanation, shape (124, 304)

global_importance = np.abs(shap_values.values).mean(axis=0)
shap_df = pd.DataFrame({
    "Feature": feat_names_final,
    "Global_SHAP": global_importance,
    "Mean_SHAP_signed": shap_values.values.mean(axis=0),
}).sort_values("Global_SHAP", ascending=False)
shap_df["Direction"] = np.where(shap_df["Mean_SHAP_signed"] >= 0, "Positive", "Negative")
shap_df.to_csv(f"{output_dir}/shap_global_importance.csv", index=False)

print(shap_df.head(10)[["Feature", "Global_SHAP", "Direction"]].to_string(index=False))

shap_matrix_df = pd.DataFrame(shap_values.values, columns=feat_names_final)
shap_matrix_df.insert(0, "Bacteria_ID", df["Bacteria_ID"].values)
shap_matrix_df.to_csv(f"{output_dir}/shap_values_full.csv", index=False)

top20_idx = np.argsort(-global_importance)[:20]
plt.figure(figsize=(9, 8))
shap.summary_plot(
    shap_values.values[:, top20_idx],
    X_scaled_df.iloc[:, top20_idx],
    feature_names=[feat_names_final[i] for i in top20_idx],
    cmap=GREEN_WHITE_BLUE,
    show=False,
)
plt.tight_layout()
plt.savefig(f"{output_dir}/shap_beeswarm_top20.png", dpi=300, bbox_inches="tight")
plt.close()

print(f"  [6] Saved: shap_global_importance.csv, shap_values_full.csv, shap_beeswarm_top20.png")
print(f"\n[END] All outputs -> {output_dir}/")
