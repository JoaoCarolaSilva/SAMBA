import warnings, os
import numpy as np
import pandas as pd
from copy import deepcopy
from itertools import combinations
from datetime import datetime

from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import RobustScaler
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import f1_score, roc_auc_score, recall_score, precision_score
from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE
from imblearn.combine import SMOTETomek, SMOTEENN
from imblearn.pipeline import Pipeline as ImbPipeline

warnings.filterwarnings("ignore")

output_dir = f"Triplet_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
os.makedirs(output_dir, exist_ok=True)
print(f"[START] Output -> {output_dir}/\n")

###
# CONFIG
###

CONTROL_RAW = {
    "Shoot_Length_Drought": 57.5,
    "Root_Length_Drought":  60.9,
    "Shoot_Weight_Drought": 13.0,
    "Root_Weight_Drought":  10.8,
}
BENEFIT_THRESHOLD = 0.10
W_LENGTH, W_WEIGHT = 0.40, 0.60

MI_TOP_N   = 200
CV_FOLDS   = 5
RANDOM_STATE = 42

###
# DATA & TARGET
###

df = pd.read_csv("Clean_data.csv")
df = df.fillna(df.mean(numeric_only=True))

drought_length_cols = [column for column in df.columns if "Drought" in column and "Length" in column]
drought_weight_cols = [column for column in df.columns if "Drought" in column and "Weight" in column]
drought_cols        = drought_length_cols + drought_weight_cols

def classify_beneficial(row):
    """
    classify_beneficial defines which bacteria are considered beneficial based on
    their performance for each phenotypical trait when compared to the control
    """
    orig      = {col: row[col] + CONTROL_RAW[col] for col in drought_cols}
    length_votes = sum(orig[c] > CONTROL_RAW[c] * (1 + BENEFIT_THRESHOLD)
                    for c in drought_length_cols)
    weight_votes = sum(orig[c] > CONTROL_RAW[c] * (1 + BENEFIT_THRESHOLD)
                    for c in drought_weight_cols)
    beneficial_score = length_votes * W_LENGTH + weight_votes * W_WEIGHT
    return int(beneficial_score >= 0.5)

df["Target"] = df[drought_cols].apply(classify_beneficial, axis=1)

y     = df["Target"].copy()
count_positives = df["Target"].sum()
count_negatives = (df["Target"] == 0).sum()
print(f"  [DATA] {len(df)} samples | {count_positives} Beneficial / {count_negatives} Non-Beneficial")

###
# FEATURE SET
###

phenotype_cols = [column for column in df.columns if "Shoot" in column or "Root" in column]
biochem_cols   = [column for column in df.columns
                  if column not in ("Bacteria_ID", "Target")
                  and column not in phenotype_cols]

features_base = df[biochem_cols].copy()

for peg, reg in [("Biofilm_PEG",     "Biofilm"),
                 ("Auxins_PEG",      "Auxins"),
                 ("ACC_PEG",         "ACC"),
                 ("Trehalose_PEG",   "Trehalose"),
                 ("Proline_PEG",     "Proline"),
                 ("Antioxidants_PEG","Antioxidants"),
                 ("Nitrogen_PEG",    "Nitrogen"),
                 ("Potassium_PEG",   "Potassium")]:
    if peg in df.columns:
        features_base[f"StressRatio_{reg}"] = df[peg] / (df[reg] + 1e-6)

for i in range(len(biochem_cols)):
    for j in range(i + 1, len(biochem_cols)):
        features_base[f"ratio_{biochem_cols[i]}_div_{biochem_cols[j]}"] = (
            df[biochem_cols[i]] / (df[biochem_cols[j]] + 1e-6))

base_features = features_base.replace([np.inf, -np.inf], np.nan).fillna(features_base.mean())
print(f"  [BASE] {base_features.shape[1]} base features")

###
# PROBE PIPELINE
###

def make_probe(sampler=None):
    """
    make_probe builds a fast classification
    pipeline for candidate feature screening
    """
    steps = [("scaler", RobustScaler())]
    if sampler is not None:
        steps.append(("sampler", deepcopy(sampler)))
    steps.append(("clf", LogisticRegression(
        class_weight="balanced", C=0.5, max_iter=2000,
        random_state=RANDOM_STATE, solver="saga",
        penalty="elasticnet", l1_ratio=0.5)))
    return ImbPipeline(steps)

def cv_score(features, target, sampler=None, n_folds=CV_FOLDS):
    """
    cv_score scores a feature matrix via 5-fold stratified CV

    """
    cv    = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_STATE)
    p     = make_probe(sampler)
    preds = cross_val_predict(p, features, target, cv=cv)
    proba = cross_val_predict(deepcopy(p), features, target,
                               cv=cv, method="predict_proba")[:, 1]
    return {
        "F1":          f1_score(target, preds, average="macro"),
        "AUC":         roc_auc_score(target, proba),
        "Recall_1":    recall_score(target, preds, pos_label=1),
        "Precision_1": precision_score(target, preds, pos_label=1, zero_division=0),
    }

###
# SAMPLER CANDIDATES
###

sampler_candidates = {
    "None":           None,
    "SMOTE_k3":       SMOTE(k_neighbors=3, random_state=RANDOM_STATE),
    "SMOTE_k4":       SMOTE(k_neighbors=4, random_state=RANDOM_STATE),
    "BorderlineSMOTE":BorderlineSMOTE(random_state=RANDOM_STATE),
    "ADASYN":         ADASYN(random_state=RANDOM_STATE),
    "SMOTETomek":     SMOTETomek(random_state=RANDOM_STATE),
    "SMOTEENN":       SMOTEENN(random_state=RANDOM_STATE),
}

###
# TRIPLET FEATURES GENERATION
###

print(f"\n  [1] Generating triplet features from {len(biochem_cols)} biochem columns...")
raw = df[biochem_cols].copy()

triplet_names, triplet_arrays = [], []
for a, b, c in combinations(biochem_cols, 3):
    # Form 1: A * B / C
    name1 = f"trip_{a}_x_{b}_div_{c}"
    vals1 = (raw[a] * raw[b]) / (raw[c] + 1e-6)
    triplet_names.append(name1)
    triplet_arrays.append(vals1.values)
    # Form 2: A / (B * C)
    name2 = f"trip_{a}_div_{b}_x_{c}"
    vals2 = raw[a] / (raw[b] * raw[c] + 1e-6)
    triplet_names.append(name2)
    triplet_arrays.append(vals2.values)

triplet_matrix = np.column_stack(triplet_arrays)
triplet_matrix = np.where(np.isinf(triplet_matrix), np.nan, triplet_matrix)
col_means      = np.nanmean(triplet_matrix, axis=0)
inds           = np.where(np.isnan(triplet_matrix))
triplet_matrix[inds] = np.take(col_means, inds[1])

print(f"       Generated {len(triplet_names)} triplet features")

###
# MUTUAL INFORMATION PRE-FILTER
###

print(f"\n  [2] Mutual information filter → keeping top {MI_TOP_N}...")
mi_scores = mutual_info_classif(triplet_matrix, y, random_state=RANDOM_STATE)
top_mi_idx = np.argsort(mi_scores)[::-1][:MI_TOP_N]
top_mi_names  = [triplet_names[i] for i in top_mi_idx]
top_mi_matrix = triplet_matrix[:, top_mi_idx]

mi_df = pd.DataFrame({
    "Feature": triplet_names,
    "MI_Score": mi_scores
}).sort_values("MI_Score", ascending=False)
mi_df.to_csv(f"{output_dir}/mi_scores_all.csv", index=False)
print(f"       MI filter done | top MI score: {mi_scores[top_mi_idx[0]]:.4f}"
      f"  |  bottom of top-{MI_TOP_N}: {mi_scores[top_mi_idx[-1]]:.4f}")

###
# PER-SAMPLER SCREENING & N-SELECTION
###

print(f"\n  [3] Per-sampler screening & N-selection ({len(sampler_candidates)} samplers)...")

sampler_summaries = []
best_overall = None
best_overall_score = (-1.0, -1.0, -1.0)

for name, sampler in sampler_candidates.items():
    try:
        print(f"\n  -- {name} --")
        baseline = cv_score(base_features, y, sampler)
        print(f"       baseline             | F1={baseline['F1']:.4f}  AUC={baseline['AUC']:.4f}"
              f"  R1={baseline['Recall_1']:.4f}")

        cv_results = []
        for i, (tname, col_data) in enumerate(zip(top_mi_names, top_mi_matrix.T)):
            if (i + 1) % 20 == 0:
                print(f"       {i+1}/{MI_TOP_N}", end="\r")
            X_aug  = pd.concat([base_features,
                                 pd.Series(col_data, name=tname, index=base_features.index)],
                                axis=1)
            scores = cv_score(X_aug, y, sampler)
            delta  = scores["F1"] - baseline["F1"]
            cv_results.append({
                "Sampler":     name,
                "Feature":     tname,
                "F1":          scores["F1"],
                "Delta_F1":    delta,
                "AUC":         scores["AUC"],
                "Recall_1":    scores["Recall_1"],
                "Precision_1": scores["Precision_1"],
                "MI_Score":    mi_scores[top_mi_idx[i]],
            })
        print()

        cv_df = pd.DataFrame(cv_results).sort_values("Delta_F1", ascending=False)
        n_positive = (cv_df["Delta_F1"] > 0).sum()
        print(f"       Triplets that improve F1: {n_positive} / {MI_TOP_N}")

        positive_trips = cv_df[cv_df["Delta_F1"] > 0]["Feature"].tolist()
        trip_lookup = {tname: triplet_matrix[:, triplet_names.index(tname)]
                       for tname in positive_trips}

        test_sizes = [0, 10, 20, 50, 100, 200]
        test_sizes = sorted(set([n for n in test_sizes if n <= len(positive_trips)]
                                + [len(positive_trips)]))

        n_sel_results = []
        for n_add in test_sizes:
            label    = "base" if n_add == 0 else f"base+top{n_add}"
            top_cols = positive_trips[:n_add]
            if n_add > 0:
                extra   = pd.DataFrame(
                    {tname: trip_lookup[tname] for tname in top_cols},
                    index=base_features.index)
                features  = pd.concat([base_features, extra], axis=1)
            else:
                features  = base_features
            scores = cv_score(features, y, sampler)
            n_sel_results.append({
                "Sampler":          name,
                "N_Triplets_Added": n_add,
                "Total_Features":   features.shape[1],
                "Label":            label,
                "CV_F1":            scores["F1"],
                "CV_AUC":           scores["AUC"],
                "Recall_1":         scores["Recall_1"],
            })
            print(f"       {label:20s} | total={features.shape[1]:4d} | "
                  f"F1={scores['F1']:.4f}  AUC={scores['AUC']:.4f}"
                  f"  R1={scores['Recall_1']:.4f}")

        n_sel_df   = pd.DataFrame(n_sel_results).sort_values("CV_F1", ascending=False)
        best_row   = n_sel_df.iloc[0]
        n_best_add = int(best_row["N_Triplets_Added"])
        best_trips = positive_trips[:n_best_add]

        sampler_summaries.append(best_row.to_dict())

        candidate_score = (best_row["CV_F1"], best_row["Recall_1"], best_row["CV_AUC"])
        if candidate_score > best_overall_score:
            best_overall_score = candidate_score
            best_overall = {
                "sampler_name": name,
                "sampler":      sampler,
                "baseline":     baseline,
                "cv_df":        cv_df,
                "n_sel_df":     n_sel_df,
                "best_row":     best_row,
                "n_best_add":   n_best_add,
                "best_trips":   best_trips,
            }
    except Exception as e:
        print(f"       {name:20s} | FAILED: {e}")

sampler_grid_df = pd.DataFrame(sampler_summaries).sort_values("CV_F1", ascending=False)
sampler_grid_df.to_csv(f"{output_dir}/sampler_grid_summary.csv", index=False)
print(f"\n  [3] Overall winner: {best_overall['sampler_name']} + {best_overall['best_row']['Label']} "
      f"(F1={best_overall_score[0]:.4f}  R1={best_overall_score[1]:.4f}  AUC={best_overall_score[2]:.4f})")

baseline       = best_overall["baseline"]
cv_df          = best_overall["cv_df"]
n_sel_df       = best_overall["n_sel_df"]
best_row       = best_overall["best_row"]
n_best_add     = best_overall["n_best_add"]
best_trips     = best_overall["best_trips"]
best_sampler_name = best_overall["sampler_name"]

cv_df.to_csv(f"{output_dir}/triplet_cv_screening.csv", index=False)
n_sel_df.to_csv(f"{output_dir}/n_selection.csv", index=False)

print(f"\n  Top 20 by Delta_F1 ({best_sampler_name}):")
print(cv_df.head(20).to_string(index=False))

###
# REPORT GENERATION
###

print(f"\n{'='*60}")
print(f"  RESULT SUMMARY")
print(f"{'='*60}")
print(f"  Best sampler:              {best_sampler_name}")
print(f"  Baseline (284 features):  F1={baseline['F1']:.4f}")
print(f"  Best configuration:       {best_row['Label']}")
print(f"  Best CV F1:               {best_row['CV_F1']:.4f}"
      f"  (delta={best_row['CV_F1']-baseline['F1']:+.4f})")
print(f"  Best Recall:              {best_row['Recall_1']:.4f}")
print(f"  Total features at best N: {int(best_row['Total_Features'])}")

if n_best_add > 0:
    print(f"\n  Triplets to add to main script ({n_best_add}):")
    for t in best_trips:
        delta = cv_df.loc[cv_df["Feature"] == t, "Delta_F1"].values[0]
        print(f"    {t}  (delta={delta:+.4f})")
else:
    print(f"\n  No triplets improve the model — base 284 features are optimal.")

with open(f"{output_dir}/triplet_report.txt", "w", encoding="utf-8") as f:
    f.write("TRIPLET FEATURE SCREENING REPORT\n" + "="*60 + "\n\n")
    f.write(f"Best sampler: {best_sampler_name}\n\n")
    f.write(f"Baseline CV F1 (284 features): {baseline['F1']:.4f}\n\n")
    f.write(f"MI pre-filter: kept top {MI_TOP_N} of {len(triplet_names)} candidates\n\n")
    f.write(f"Sampler grid summary (best N per sampler):\n")
    f.write(sampler_grid_df.to_string(index=False) + "\n\n")
    f.write(f"Top 30 triplets by Delta_F1 (5-fold CV, {best_sampler_name}):\n")
    f.write(cv_df.head(30).to_string(index=False) + "\n\n")
    f.write(f"N-selection results ({best_sampler_name}):\n")
    f.write(n_sel_df.to_string(index=False) + "\n\n")
    f.write(f"RECOMMENDATION: add {n_best_add} triplets "
            f"-> {int(best_row['Total_Features'])} total features\n")
    if n_best_add > 0:
        f.write("Features to add:\n")
        for t in best_trips:
            f.write(f"  {t}\n")

print(f"\n[END] All outputs -> {output_dir}/")