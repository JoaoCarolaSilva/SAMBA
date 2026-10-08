import warnings, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations
from datetime import datetime
from scipy.spatial.distance import cdist

warnings.filterwarnings("ignore")

###
# CONFIG
###

CLASSIFIER_DIR        = "Classifier_results_20260920_115035"
DATA_PATH             = "Clean_data.csv"
TRIPLET_SCREENING_CSV = "Triplet_results_20260917_231854/triplet_cv_screening.csv"
TRIPLET_N             = 17

K            = 5      # strains per SynCom
N_SYNCOMS    = 3      # SynComs per approach
MIN_DIFF     = 2      # min strains differing between any two SynComs
P_THRESHOLD  = 0.5
DIV_FEATURES = 20
DIV_WEIGHT   = 0.5
RANDOM_SEED  = 42

CONTROL_RAW = {
    "Shoot_Length_Drought": 57.5,
    "Root_Length_Drought":  60.9,
    "Shoot_Weight_Drought": 13.0,
    "Root_Weight_Drought":  10.8,
}
W_LENGTH, W_WEIGHT = 0.40, 0.60

BINARY_TRAITS   = ["Glycine_Betaine", "Silica"]
PRESENCE_TRAITS = ["Manganese", "Calcium", "Polyamines", "Zinc"]

COVERAGE_TRAITS = [
    "Biofilm_PEG", "Auxins_PEG", "ACC_PEG", "Trehalose_PEG",
    "Proline_PEG", "Antioxidants_PEG", "Potassium_PEG", "Phosphate",
    "Glycine_Betaine", "Silica", "Manganese", "Calcium", "Polyamines", "Zinc"
]

TRAIT_LABELS = {
    "Biofilm_PEG":"Biofilm\n(PEG)", "Auxins_PEG":"Auxins\n(PEG)",
    "ACC_PEG":"ACC\n(PEG)", "Trehalose_PEG":"Trehalose\n(PEG)",
    "Proline_PEG":"Proline\n(PEG)", "Antioxidants_PEG":"Antioxid.\n(PEG)",
    "Potassium_PEG":"Potassium\n(PEG)", "Phosphate":"Phosphate",
    "Glycine_Betaine":"Glycine\nBetaine", "Silica":"Silica",
    "Manganese":"Manganese", "Calcium":"Calcium",
    "Polyamines":"Polyamines", "Zinc":"Zinc",
}

output_dir = f"syncom_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
os.makedirs(output_dir, exist_ok=True)
print(f"[START] SynCom Designer -> {output_dir}/\n")

###
# LOAD CLASSIFIER OUTPUTS
###

import joblib

ranking_df    = pd.read_csv(os.path.join(CLASSIFIER_DIR, "bacteria_ranking.csv"))
importance_df = pd.read_csv(os.path.join(CLASSIFIER_DIR, "feature_importance.csv")
                            ).sort_values("Importance", ascending=False)

feat_names_final = joblib.load(os.path.join(CLASSIFIER_DIR, "feature_names.pkl"))
print(f"  [MODEL] Loaded {len(feat_names_final)} features from classifier")

candidates = ranking_df[
    (ranking_df["Predicted"] == "Beneficial") &
    (ranking_df["P_Beneficial"] >= P_THRESHOLD)
].copy().reset_index(drop=True)

print(f"  [DATA] {len(ranking_df)} total strains")
print(f"  [CANDIDATES A-D] {len(candidates)} predicted Beneficial (P >= {P_THRESHOLD})")

###
# LOAD RAW DATA & RECONSTRUCT FEATURES
###

df = pd.read_csv(DATA_PATH)
df = df.fillna(df.mean(numeric_only=True))
df.index = df["Bacteria_ID"].values

drought_cols        = list(CONTROL_RAW.keys())
drought_length_cols = [c for c in CONTROL_RAW if "Length" in c]
drought_weight_cols = [c for c in CONTROL_RAW if "Weight" in c]

def compute_drought_score(row):
    """
    compute_drought_score takes the phenotype trait values
    and calculates the drought score of each strain
    """
    orig      = {col: row[col] + CONTROL_RAW[col] for col in drought_cols}
    len_score = np.mean([(orig[c]-CONTROL_RAW[c])/CONTROL_RAW[c] for c in drought_length_cols])
    wgt_score = np.mean([(orig[c]-CONTROL_RAW[c])/CONTROL_RAW[c] for c in drought_weight_cols])
    return W_LENGTH * len_score + W_WEIGHT * wgt_score

df["Drought_Score"] = df[drought_cols].apply(compute_drought_score, axis=1)

phenotype_cols = [c for c in df.columns if "Shoot" in c or "Root" in c]
all_biochem    = [c for c in df.columns
                  if c not in ("Bacteria_ID",) and c not in phenotype_cols]

X = df[all_biochem].copy()

for peg, reg in [("Biofilm_PEG","Biofilm"),("Auxins_PEG","Auxins"),
                 ("ACC_PEG","ACC"),("Trehalose_PEG","Trehalose"),
                 ("Proline_PEG","Proline"),("Antioxidants_PEG","Antioxidants"),
                 ("Nitrogen_PEG","Nitrogen"),("Potassium_PEG","Potassium")]:
    if peg in df.columns:
        X[f"StressRatio_{reg}"] = df[peg] / (df[reg] + 1e-6)

for i in range(len(all_biochem)):
    for j in range(i+1, len(all_biochem)):
        X[f"ratio_{all_biochem[i]}_div_{all_biochem[j]}"] = (
            df[all_biochem[i]] / (df[all_biochem[j]] + 1e-6))

def add_triplets(X_df, raw_df, csv_path, n_top):
    """
    add_triplets adds the previously generated triplet features
    to the feature matrix
    """
    if not os.path.exists(csv_path):
        print(f"  [TRIPLETS] Not found — skipping"); return X_df
    trip_df   = pd.read_csv(csv_path)
    positives = trip_df[trip_df["Delta_F1"] > 0].head(n_top)
    added = 0
    for _, row in positives.iterrows():
        name = row["Feature"]; inner = name[len("trip_"):]
        if "_x_" in inner.split("_div_")[0]:
            ab, c = inner.split("_div_",1); a, b = ab.split("_x_",1)
            if all(col in raw_df.columns for col in [a,b,c]):
                X_df[name] = (raw_df[a]*raw_df[b])/(raw_df[c]+1e-6); added+=1
        else:
            a, bc = inner.split("_div_",1); b, c = bc.split("_x_",1)
            if all(col in raw_df.columns for col in [a,b,c]):
                X_df[name] = raw_df[a]/(raw_df[b]*raw_df[c]+1e-6); added+=1
    print(f"  [TRIPLETS] {added} added"); return X_df

X = add_triplets(X, df, TRIPLET_SCREENING_CSV, TRIPLET_N)
X = X.replace([np.inf,-np.inf], np.nan).fillna(X.mean())
X.index = df["Bacteria_ID"].values


missing = [f for f in feat_names_final if f not in X.columns]
if missing:
    print(f"  [WARNING] {len(missing)} classifier features not found in reconstructed matrix")
X = X[[f for f in feat_names_final if f in X.columns]]
print(f"  [FEATURES] {X.shape[1]} features (matching classifier feature set)\n")


cand_ids     = candidates["Bacteria_ID"].values
X_candidates = X.reindex(cand_ids)
p_scores     = candidates.set_index("Bacteria_ID")["P_Beneficial"]
n_cand       = len(candidates)


all_ids      = df["Bacteria_ID"].values
n_all        = len(all_ids)

###
# COVERAGE THRESHOLDS
###

def compute_coverage_thresholds(df, traits):
    """
    compute_coverage_threshold defines when a bacterial strain
    counts as "covering" a certain trait
    """
    thresholds = {}
    for col in traits:
        vals = df[col]
        if col in BINARY_TRAITS:
            thresholds[col] = ("binary", 1.0)
        elif col in PRESENCE_TRAITS:
            thresholds[col] = ("presence", 0.0)
        else:
            pos_vals = vals[vals > 0]
            if len(pos_vals) == 0:
                thresholds[col] = ("excluded", None)
            else:
                pos_mean = pos_vals.mean()
                pos_sd   = pos_vals.std() if len(pos_vals) > 1 else 0.0
                thresholds[col] = ("pos_mean_sd", pos_mean + 0.5 * pos_sd)
    return thresholds

thresholds = compute_coverage_thresholds(df, COVERAGE_TRAITS)

def strain_covers(strain_vals, col, thresholds):
    """
    strain_covers evaluates whether a
    strain covers a trait based on the threshold
    defined in compute_coverage_thresholds
    """
    rule, thresh = thresholds[col]
    val = strain_vals[col]
    if rule == "binary":   return int(val == 1)
    elif rule == "presence": return int(val > 0)
    elif rule == "pos_mean_sd": return int(val > thresh)
    else: return 0

covered_matrix = np.zeros((n_cand, len(COVERAGE_TRAITS)), dtype=int)
for i, sid in enumerate(cand_ids):
    for j, col in enumerate(COVERAGE_TRAITS):
        covered_matrix[i, j] = strain_covers(df.loc[sid], col, thresholds)

print(f"  [COVERAGE] {len(COVERAGE_TRAITS)} biological traits | "
      f"{len(thresholds)} thresholds computed")

###
# FEATURE SPACES
###

avail_feats  = [f for f in importance_df["Feature"] if f in X.columns]
div_features = avail_feats[:DIV_FEATURES]
eng_features = avail_feats[:10]

X_div     = X_candidates[div_features].copy()
X_div     = (X_div - X_div.min()) / (X_div.max() - X_div.min() + 1e-9)
X_div_arr = X_div.values
dist_matrix = cdist(X_div_arr, X_div_arr, metric="euclidean")

eng_medians     = X[eng_features].median()
eng_covered_mat = (X_candidates[eng_features] > eng_medians).values.astype(int)

###
# SCORE ALL COMBINATIONS (A,B,C,D)
###

print(f"\n  [SCORING] Evaluating C({n_cand},{K}) = ", end="")
all_combos = list(combinations(range(n_cand), K))
n_combos   = len(all_combos)
combos_arr = np.array(all_combos)
print(f"{n_combos:,} combinations...")

p_arr   = np.array([p_scores[cand_ids[i]] for i in range(n_cand)])
score_A = p_arr[combos_arr].mean(axis=1)

div_score_arr = np.array([
    np.mean([dist_matrix[i,j] for i,j in combinations(c,2)])
    for c in all_combos])

eng_coverage_arr = np.array([
    eng_covered_mat[list(c),:].any(axis=0).sum() for c in all_combos])

coverage_arr = np.array([
    covered_matrix[list(c),:].any(axis=0).sum() for c in all_combos])

score_A_norm   = (score_A-score_A.min())/(score_A.max()-score_A.min()+1e-9)
div_norm       = (div_score_arr-div_score_arr.min())/(div_score_arr.max()-div_score_arr.min()+1e-9)
eng_cov_norm   = (eng_coverage_arr-eng_coverage_arr.min())/(eng_coverage_arr.max()-eng_coverage_arr.min()+1e-9)
cov_norm       = (coverage_arr-coverage_arr.min())/(coverage_arr.max()-coverage_arr.min()+1e-9)

score_B = eng_cov_norm + div_norm*0.01 + score_A_norm*0.001
score_C = DIV_WEIGHT*div_norm + (1-DIV_WEIGHT)*score_A_norm
score_D = cov_norm + div_norm*0.01 + score_A_norm*0.001

print(f"  [SCORING] Done.\n")

###
# GREEDY SELECTION
###

def greedy_select(scores, n_select, min_diff, forbidden_sets=None):
    """
    greedy_select picks up to n_select feature
    combinations from a ranked list
    """
    ranked_idx    = np.argsort(scores)[::-1]
    selected      = []
    selected_sets = []
    for idx in ranked_idx:
        combo = set(combos_arr[idx])
        fs    = frozenset(combo)
        # intra-approach diversity
        intra_ok = all(len(combo-prev) >= min_diff for prev in selected_sets)
        # inter-approach: reject if identical to any already selected across A/B/C/D
        inter_ok = (forbidden_sets is None) or (fs not in forbidden_sets)
        if intra_ok and inter_ok:
            selected.append(idx)
            selected_sets.append(combo)
        if len(selected) == n_select:
            break
    return selected


used_sets = set()

selected_A = greedy_select(score_A, N_SYNCOMS, MIN_DIFF, used_sets)
used_sets.update(frozenset(combos_arr[i]) for i in selected_A)

selected_B = greedy_select(score_B, N_SYNCOMS, MIN_DIFF, used_sets)
used_sets.update(frozenset(combos_arr[i]) for i in selected_B)

selected_C = greedy_select(score_C, N_SYNCOMS, MIN_DIFF, used_sets)
used_sets.update(frozenset(combos_arr[i]) for i in selected_C)

selected_D = greedy_select(score_D, N_SYNCOMS, MIN_DIFF, used_sets)

###
# APPROACH E
###

print("  [APPROACH E] Scoring all C(124,5) combinations by actual drought score...")
all_combos_E = list(combinations(range(n_all), K))
combos_arr_E = np.array(all_combos_E)
score_arr    = df["Drought_Score"].values
score_E      = score_arr[combos_arr_E].mean(axis=1)

def greedy_select_E(scores, combos_arr, n_select, min_diff):
    """
    greedy_select_E works similarly to greedy_select
    """
    ranked_idx = np.argsort(scores)[::-1]
    selected, selected_sets = [], []
    for idx in ranked_idx:
        combo = set(combos_arr[idx])
        if all(len(combo-prev) >= min_diff for prev in selected_sets):
            selected.append(idx); selected_sets.append(combo)
        if len(selected) == n_select:
            break
    return selected

selected_E = greedy_select_E(score_E, combos_arr_E, N_SYNCOMS, MIN_DIFF)
print(f"  [APPROACH E] Done. {len(list(combinations(range(n_all),K))):,} combinations scored.")

###
# RANDOM CONTROL
###

print("  [RANDOM] Selecting random SynComs from all 124 strains...")
rng = np.random.default_rng(RANDOM_SEED)
random_selected = []
random_sets     = []
attempts        = 0

while len(random_selected) < N_SYNCOMS and attempts < 100000:
    combo = frozenset(rng.choice(n_all, K, replace=False))
    if all(len(combo - prev) >= MIN_DIFF for prev in random_sets):
        random_selected.append(list(combo))
        random_sets.append(combo)
    attempts += 1

print(f"  [RANDOM] {len(random_selected)} SynComs selected.\n")

###
# FORMAT RESULTS
###

def get_strain_drought_score(sid):
    """
    get_strain_drought_score returns a strains drought score
    """
    try:
        return round(df.loc[sid, "Drought_Score"], 4)
    except:
        return None

def format_results_abcd(selected_indices, label, score_arr_used):
    """
    format_results_abcd builds the results table for approaches A, B, C, D
    """
    rows = []
    for rank, idx in enumerate(selected_indices, 1):
        combo = combos_arr[idx]; strains = [cand_ids[i] for i in combo]
        p_vals = [p_arr[i] for i in combo]
        row = {"Rank": rank, "Approach": label,
               "SynCom_ID":            f"{label}_SC{rank}",
               "Strains":              " | ".join(str(s) for s in strains),
               "Mean_P_Beneficial":    round(np.mean(p_vals), 4),
               "Min_P_Beneficial":     round(np.min(p_vals), 4),
               "Diversity_Score":      round(div_score_arr[idx], 4),
               "Bio_Trait_Coverage":   f"{coverage_arr[idx]}/{len(COVERAGE_TRAITS)}",
               "Eng_Feature_Coverage": f"{eng_coverage_arr[idx]}/{len(eng_features)}",
               "Score_Used":           round(score_arr_used[idx], 4)}
        for s_rank,(sid,sp) in enumerate(zip(strains,p_vals),1):
            row[f"Strain_{s_rank}"] = sid; row[f"P_Ben_{s_rank}"] = round(sp,4)
        rows.append(row)
    return pd.DataFrame(rows)

def format_results_E(selected_indices):
    """
    format_results_E builds the results table for approach E
    """
    rows = []
    for rank, idx in enumerate(selected_indices, 1):
        combo   = combos_arr_E[idx]
        strains = [all_ids[i] for i in combo]
        d_vals  = [score_arr[i] for i in combo]
        row = {"Rank": rank, "Approach": "E_Phenotype",
               "SynCom_ID":         f"E_Phenotype_SC{rank}",
               "Strains":           " | ".join(str(s) for s in strains),
               "Mean_Drought_Score": round(np.mean(d_vals), 4),
               "Min_Drought_Score":  round(np.min(d_vals), 4),
               "Score_Used":         round(score_E[idx], 4)}
        for s_rank,(sid,dv) in enumerate(zip(strains,d_vals),1):
            row[f"Strain_{s_rank}"]       = sid
            row[f"DroughtScore_{s_rank}"] = round(dv, 4)
        rows.append(row)
    return pd.DataFrame(rows)

def format_results_random(random_combos):
    """
    format_results_random builds the results table for approach F
    """
    rows = []
    for rank, combo in enumerate(random_combos, 1):
        strains = [all_ids[i] for i in combo]
        d_vals  = [score_arr[i] for i in combo]
        p_vals  = []
        for sid in strains:
            p_row = ranking_df[ranking_df["Bacteria_ID"].astype(str) == str(sid)]
            p_vals.append(p_row["P_Beneficial"].values[0] if len(p_row) > 0 else None)
        row = {"Rank": rank, "Approach": "F_Random",
               "SynCom_ID":          f"F_Random_SC{rank}",
               "Strains":            " | ".join(str(s) for s in strains),
               "Mean_Drought_Score": round(np.mean(d_vals), 4),
               "Score_Used":         "random"}
        for s_rank,(sid,dv,pv) in enumerate(zip(strains,d_vals,p_vals),1):
            row[f"Strain_{s_rank}"]       = sid
            row[f"DroughtScore_{s_rank}"] = round(dv, 4)
            row[f"P_Ben_{s_rank}"]        = round(pv,4) if pv is not None else None
        rows.append(row)
    return pd.DataFrame(rows)

results_A = format_results_abcd(selected_A, "A_Elite",         score_A)
results_B = format_results_abcd(selected_B, "B_FeatCoverage",  score_B)
results_C = format_results_abcd(selected_C, "C_Balanced",      score_C)
results_D = format_results_abcd(selected_D, "D_TraitCoverage", score_D)
results_E = format_results_E(selected_E)
results_R = format_results_random(random_selected)

all_results = pd.concat([results_A,results_B,results_C,results_D,
                          results_E,results_R], ignore_index=True)
all_results.to_csv(f"{output_dir}/all_approaches_summary.csv", index=False)
for res, fname in [(results_A,"approach_A_elite"),(results_B,"approach_B_feat_coverage"),
                   (results_C,"approach_C_balanced"),(results_D,"approach_D_trait_coverage"),
                   (results_E,"approach_E_phenotype"),(results_R,"approach_F_random")]:
    res.to_csv(f"{output_dir}/{fname}.csv", index=False)

###
# REPORT
###

with open(f"{output_dir}/syncom_report.txt","w",encoding="utf-8") as f:
    f.write("="*70+"\nSynCom Designer Report — Final Version\n"+"="*70+"\n\n")
    f.write(f"Classifier:            {CLASSIFIER_DIR}\n")
    f.write(f"Candidates (A-D):      {n_cand} predicted Beneficial\n")
    f.write(f"Pool (E, Random):      {n_all} all strains\n")
    f.write(f"SynCom size (k):       {K}\n")
    f.write(f"SynComs/approach:      {N_SYNCOMS}\n")
    f.write(f"Min strain diff:       {MIN_DIFF}\n")
    f.write(f"Inter-approach unique: Yes (A/B/C/D)\n")
    f.write(f"Total combinations:    {n_combos:,} (A-D) | "
            f"{len(list(combinations(range(n_all),K))):,} (E)\n\n")

    for res_df, label in [(results_A,"A — Elite"),(results_B,"B — Feature Coverage"),
                           (results_C,"C — Balanced"),(results_D,"D — Trait Coverage")]:
        f.write(f"\n{'─'*60}\nApproach {label}\n{'─'*60}\n")
        for _, row in res_df.iterrows():
            f.write(f"  {row['SynCom_ID']}:  {row['Strains']}\n")
            f.write(f"    Mean P={row['Mean_P_Beneficial']:.4f}  "
                    f"Min P={row['Min_P_Beneficial']:.4f}  "
                    f"Div={row['Diversity_Score']:.4f}  "
                    f"Bio={row['Bio_Trait_Coverage']}  Eng={row['Eng_Feature_Coverage']}\n")

    f.write(f"\n{'─'*60}\nApproach E — Phenotype-informed (positive control)\n{'─'*60}\n")
    for _, row in results_E.iterrows():
        f.write(f"  {row['SynCom_ID']}:  {row['Strains']}\n")
        f.write(f"    Mean Drought Score={row['Mean_Drought_Score']:.4f}  "
                f"Min={row['Min_Drought_Score']:.4f}\n")

    f.write(f"\n{'─'*60}\nApproach F — Random (negative control)\n{'─'*60}\n")
    for _, row in results_R.iterrows():
        f.write(f"  {row['SynCom_ID']}:  {row['Strains']}\n")
        f.write(f"    Mean Drought Score={row['Mean_Drought_Score']:.4f}\n")

print(f"\n{'='*60}\n  SYNCOM DESIGNER — RESULTS SUMMARY\n{'='*60}")
for res_df, label in [(results_A,"A — Elite"),(results_B,"B — Feature Coverage"),
                       (results_C,"C — Balanced"),(results_D,"D — Trait Coverage")]:
    print(f"\n  Approach {label}:")
    for _, row in res_df.iterrows():
        print(f"    {row['SynCom_ID']}: [{row['Strains']}]  "
              f"P={row['Mean_P_Beneficial']:.3f}  "
              f"Bio={row['Bio_Trait_Coverage']}  Eng={row['Eng_Feature_Coverage']}")

print(f"\n  Approach E — Phenotype (positive control):")
for _, row in results_E.iterrows():
    print(f"    {row['SynCom_ID']}: [{row['Strains']}]  "
          f"DroughtScore={row['Mean_Drought_Score']:.3f}")

print(f"\n  Approach F — Random (negative control):")
for _, row in results_R.iterrows():
    print(f"    {row['SynCom_ID']}: [{row['Strains']}]  "
          f"DroughtScore={row['Mean_Drought_Score']:.3f}")

print(f"\n[END] All outputs -> {output_dir}/")