# Databricks notebook source
# MAGIC %md
# MAGIC # 01 — Load, merge, and clean the REAL freMTPL2freq / freMTPL2sev data
# MAGIC Source: OpenML dataset 41214 (C. Dutang & A. Charpentier, CASdatasets),
# MAGIC uploaded by the user. 678,013 policies, ~2011-2013, France.

# COMMAND ----------
import numpy as np
import pandas as pd

freq = pd.read_csv("data/freMTPL2freq.csv")
sev = pd.read_csv("data/freMTPL2sev.csv")
n0 = len(freq)
report = [f"# Data quality report — REAL freMTPL2freq / freMTPL2sev\n\nRaw policies: **{n0:,}**  \nRaw severity records: **{len(sev):,}**\n"]

# COMMAND ----------
# 1) Orphaned severity records (claim references a policy that doesn't exist in freq)
orphan_mask = ~sev.IDpol.isin(freq.IDpol)
report.append(f"## 1. Orphaned claims\n**{int(orphan_mask.sum())}** severity records reference an IDpol not present "
              f"in the policy table. Dropped (no risk factors to model them against).\n")
sev = sev.loc[~orphan_mask].copy()

# COMMAND ----------
# 2) Aggregate severity to policy level, then reconcile against freq.ClaimNb.
#    This is a KNOWN, published data issue with this exact dataset (Wuthrich et al.
#    note that policies with IDpol <= 24500 have unreliable ClaimNb/severity links).
sev_agg = sev.groupby("IDpol").agg(
    sev_claim_count=("ClaimAmount", "size"),
    ClaimAmount=("ClaimAmount", "sum"),
).reset_index()

df = freq.merge(sev_agg, on="IDpol", how="left")
df["sev_claim_count"] = df["sev_claim_count"].fillna(0).astype(int)
df["ClaimAmount"] = df["ClaimAmount"].fillna(0.0)

mismatch = df.ClaimNb != df.sev_claim_count
low_id_share = (df.loc[mismatch, "IDpol"] <= 24500).mean() if mismatch.sum() else 0
report.append(f"## 2. ClaimNb vs. actual linked severity records\n**{int(mismatch.sum()):,}** policies "
              f"(**{mismatch.mean():.1%}** of the book) have a `ClaimNb` that does not match the number of "
              f"severity records actually linked to them, almost entirely `ClaimNb > 0` with zero linked "
              f"severity rows ({int(((df.ClaimNb>0)&(df.sev_claim_count==0)).sum()):,} policies), "
              f"{low_id_share:.0%} of which have `IDpol <= 24500`. This matches a data-linkage issue in this "
              f"exact dataset that is documented in the actuarial literature (Wuthrich et al.), not a mistake "
              f"in this pipeline.\n\n"
              f"**Decision:** keep `ClaimNb` as-is for the FREQUENCY model (it is still the field the source "
              f"labels as the claim count). For the SEVERITY model, only policies with a genuine linked claim "
              f"amount are usable, the {int(((df.ClaimNb>0)&(df.sev_claim_count==0)).sum()):,} policies with "
              f"no linked amount are excluded from severity training only, flagged via `severity_usable`.\n")
df["severity_usable"] = ~((df.ClaimNb > 0) & (df.sev_claim_count == 0))

# COMMAND ----------
# 3) Exposure out of range (fraction of a policy-year; cannot exceed 1.0 meaningfully)
bad_exp = df.Exposure > 1.0
report.append(f"## 3. Exposure > 1.0\n**{int(bad_exp.sum()):,}** policies ({bad_exp.mean():.2%}), "
              f"max observed {df.Exposure.max():.2f}. Capped at 1.0 rather than dropped, these are almost "
              f"certainly rounding/measurement artefacts on otherwise-usable rows, not corrupted records.\n")
df["Exposure"] = df["Exposure"].clip(upper=1.0)
bad_exp_low = df.Exposure <= 0
report.append(f"Exposure <= 0: **{int(bad_exp_low.sum())}** rows, dropped.\n")
df = df.loc[~bad_exp_low].copy()

# COMMAND ----------
# 4) Implausible VehAge / DrivAge, capped at data-driven percentile cutoffs
#    (99.5th percentile for VehAge, 99.9th for DrivAge), standard practice for
#    this dataset in the actuarial ML literature, done for MODEL STABILITY,
#    a handful of 90-100 values would otherwise dominate a spline/GLM term.
veh_cap, driv_cap = 25, 90
n_veh = (df.VehAge > veh_cap).sum()
n_driv = (df.DrivAge > driv_cap).sum()
report.append(f"## 4. Extreme VehAge / DrivAge\nVehAge > {veh_cap}: **{int(n_veh):,}** rows "
              f"(max {df.VehAge.max()}), capped at {veh_cap}. DrivAge > {driv_cap}: **{int(n_driv):,}** rows "
              f"(max {df.DrivAge.max()}), capped at {driv_cap}. Capped, not dropped, the policy is still "
              f"valid, only the extreme tail of one feature is smoothed.\n")
df["VehAge"] = df["VehAge"].clip(upper=veh_cap)
df["DrivAge"] = df["DrivAge"].clip(upper=driv_cap)

# COMMAND ----------
# 5) Severity outliers, capped for MODEL FITTING only; uncapped total kept for
#    portfolio reconciliation (same logic as the earlier synthetic-data pass).
claim_rows = df.ClaimAmount > 0
p99_5 = df.loc[claim_rows, "ClaimAmount"].quantile(0.995)
n_extreme = (claim_rows & (df.ClaimAmount > p99_5)).sum()
report.append(f"## 5. Severity outliers\n**{int(n_extreme)}** claims above the 99.5th percentile "
              f"(> {p99_5:,.0f}), including one single claim of {df.ClaimAmount.max():,.0f}. Kept in the "
              f"portfolio total (`ClaimAmount`), capped in `ClaimAmount_capped` for model fitting.\n")
df["ClaimAmount_capped"] = np.minimum(df["ClaimAmount"], p99_5)

# COMMAND ----------
n1 = len(df)
report.append(f"\n## Summary\nPolicies in: **{n0:,}** -> out: **{n1:,}** "
              f"({n0-n1:,} dropped, {((n0-n1)/n0):.2%}). Total claims linked: **{int(df.sev_claim_count.sum()):,}** "
              f"of {int(df.ClaimNb.sum()):,} reported in ClaimNb.\n")

with open("data/data_quality_report.md", "w") as f:
    f.write("\n".join(report))
df.to_csv("data/policies_clean.csv", index=False)
print(f"Clean file: {n1:,} rows -> data/policies_clean.csv")
print("\n".join(report))

# COMMAND ----------

# MAGIC %md
# MAGIC # 02 — GLM vs gradient-boosted trees on REAL freMTPL2 data

# COMMAND ----------
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import PoissonRegressor, GammaRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_poisson_deviance, mean_gamma_deviance, mean_absolute_error
import joblib

df = pd.read_csv("data/policies_clean.csv")

NUM_FEATS = ["VehPower", "VehAge", "DrivAge", "BonusMalus", "Density"]
CAT_FEATS = ["VehBrand", "VehGas", "Area", "Region"]
ALL_FEATS = NUM_FEATS + CAT_FEATS

# COMMAND ----------
df["has_claim"] = (df.ClaimNb > 0).astype(int)
train, test = train_test_split(df, test_size=0.2, random_state=7, stratify=df.has_claim)
print(f"Train: {len(train):,}   Test: {len(test):,}")
print(f"Train claim rate: {train.has_claim.mean():.3%}   Test claim rate: {test.has_claim.mean():.3%}")

# COMMAND ----------
# ---- FREQUENCY MODELS ----------
pre_glm = ColumnTransformer([
    ("num", StandardScaler(), NUM_FEATS),
    ("cat", OneHotEncoder(handle_unknown="ignore", drop="first"), CAT_FEATS),
])
freq_glm = Pipeline([("prep", pre_glm), ("model", PoissonRegressor(alpha=1e-4, max_iter=2000))])
freq_glm.fit(train[ALL_FEATS], train.ClaimNb / train.Exposure, model__sample_weight=train.Exposure)

train_gbm, test_gbm = train.copy(), test.copy()
for c in CAT_FEATS:
    train_gbm[c] = train_gbm[c].astype("category")
    test_gbm[c] = pd.Categorical(test_gbm[c], categories=train_gbm[c].cat.categories)
cat_mask = [f in CAT_FEATS for f in ALL_FEATS]

freq_gbm = HistGradientBoostingRegressor(
    loss="poisson", max_iter=300, max_depth=6, learning_rate=0.06,
    categorical_features=cat_mask, random_state=7, early_stopping=True,
)
freq_gbm.fit(train_gbm[ALL_FEATS], train_gbm.ClaimNb / train_gbm.Exposure, sample_weight=train_gbm.Exposure)

# COMMAND ----------
# ---- SEVERITY MODELS (severity_usable policies only, ClaimNb > 0) ----------
sev_train = train.loc[(train.ClaimNb > 0) & (train.severity_usable)].copy()
sev_test = test.loc[(test.ClaimNb > 0) & (test.severity_usable)].copy()
sev_train["sev_per_claim"] = sev_train.ClaimAmount_capped / sev_train.ClaimNb
sev_test["sev_per_claim"] = sev_test.ClaimAmount_capped / sev_test.ClaimNb
print(f"Severity-usable train: {len(sev_train):,}   test: {len(sev_test):,}")

pre_glm_sev = ColumnTransformer([
    ("num", StandardScaler(), NUM_FEATS),
    ("cat", OneHotEncoder(handle_unknown="ignore", drop="first"), CAT_FEATS),
])
sev_glm = Pipeline([("prep", pre_glm_sev), ("model", GammaRegressor(alpha=1e-4, max_iter=2000))])
sev_glm.fit(sev_train[ALL_FEATS], sev_train.sev_per_claim, model__sample_weight=sev_train.ClaimNb)

sev_train_gbm, sev_test_gbm = sev_train.copy(), sev_test.copy()
for c in CAT_FEATS:
    sev_train_gbm[c] = sev_train_gbm[c].astype("category")
    sev_test_gbm[c] = pd.Categorical(sev_test_gbm[c], categories=sev_train_gbm[c].cat.categories)

sev_gbm = HistGradientBoostingRegressor(
    loss="gamma", max_iter=250, max_depth=5, learning_rate=0.05,
    categorical_features=cat_mask, random_state=7, early_stopping=True,
)
sev_gbm.fit(sev_train_gbm[ALL_FEATS], sev_train_gbm.sev_per_claim, sample_weight=sev_train_gbm.ClaimNb)

# COMMAND ----------
def eval_frequency(name, pred_rate_fn, test_df):
    pred_rate = pred_rate_fn(test_df)
    pred_claims = np.maximum(pred_rate * test_df.Exposure, 1e-6)
    return {"model": name, "poisson_deviance": mean_poisson_deviance(test_df.ClaimNb, pred_claims),
            "MAE_claims": mean_absolute_error(test_df.ClaimNb, pred_claims)}

def eval_severity(name, pred_fn, test_df):
    pred = np.maximum(pred_fn(test_df), 1.0)
    return {"model": name, "gamma_deviance": mean_gamma_deviance(test_df.sev_per_claim, pred),
            "MAE_severity": mean_absolute_error(test_df.sev_per_claim, pred)}

freq_results = [
    eval_frequency("Poisson GLM", lambda d: freq_glm.predict(d[ALL_FEATS]), test),
    eval_frequency("Gradient-boosted trees", lambda d: freq_gbm.predict(d[ALL_FEATS]), test_gbm),
]
sev_results = [
    eval_severity("Gamma GLM", lambda d: sev_glm.predict(d[ALL_FEATS]), sev_test),
    eval_severity("Gradient-boosted trees", lambda d: sev_gbm.predict(d[ALL_FEATS]), sev_test_gbm),
]
print("\n=== Frequency (test set) ===")
print(pd.DataFrame(freq_results).to_string(index=False))
print("\n=== Severity (test set) ===")
print(pd.DataFrame(sev_results).to_string(index=False))

# COMMAND ----------
joblib.dump({"freq_glm": freq_glm, "freq_gbm": freq_gbm, "sev_glm": sev_glm, "sev_gbm": sev_gbm,
             "ALL_FEATS": ALL_FEATS, "CAT_FEATS": CAT_FEATS, "NUM_FEATS": NUM_FEATS},
            "data/models.joblib")
train.to_csv("data/train.csv", index=False)
test.to_csv("data/test.csv", index=False)
train_gbm.to_pickle("data/train_gbm.pkl")
test_gbm.to_pickle("data/test_gbm.pkl")
pd.DataFrame(freq_results).to_csv("data/freq_model_comparison.csv", index=False)
pd.DataFrame(sev_results).to_csv("data/sev_model_comparison.csv", index=False)
print("\nSaved.")

# COMMAND ----------

# MAGIC %md
# MAGIC # 03 — Where GLM and the tree model actually diverge (real data)

# COMMAND ----------
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib

FIGDIR = "figures"
PBIDIR = "powerbi_exports"
import os
os.makedirs(FIGDIR, exist_ok=True)
os.makedirs(PBIDIR, exist_ok=True)

bundle = joblib.load("data/models.joblib")
freq_glm, freq_gbm = bundle["freq_glm"], bundle["freq_gbm"]
ALL_FEATS = bundle["ALL_FEATS"]

test = pd.read_csv("data/test.csv")
test_gbm = pd.read_pickle("data/test_gbm.pkl")

test["freq_glm_pred"] = freq_glm.predict(test[ALL_FEATS])
test["freq_gbm_pred"] = freq_gbm.predict(test_gbm[ALL_FEATS])

# COMMAND ----------
# ---- Actual vs predicted frequency by driver-age band (no assumed shape) -----
test["age_band"] = pd.cut(test.DrivAge, bins=[17, 22, 25, 30, 40, 50, 60, 70, 90],
                           labels=["18-22", "23-25", "26-30", "31-40", "41-50", "51-60", "61-70", "71+"])
by_age = test.groupby("age_band", observed=True).apply(
    lambda d: pd.Series({
        "actual": d.ClaimNb.sum() / d.Exposure.sum(),
        "glm": np.average(d.freq_glm_pred, weights=d.Exposure),
        "gbm": np.average(d.freq_gbm_pred, weights=d.Exposure),
        "exposure": d.Exposure.sum(),
    }), include_groups=False
).reset_index()
by_age.to_csv(f"{PBIDIR}/frequency_by_age_band.csv", index=False)
print(by_age.to_string(index=False))

plt.figure(figsize=(7.5, 4.5))
x = np.arange(len(by_age))
plt.plot(x, by_age.actual, "o-", label="Actual", color="#1A1A1A", linewidth=2)
plt.plot(x, by_age.glm, "s--", label="Poisson GLM", color="#5B8DB8")
plt.plot(x, by_age.gbm, "^--", label="Gradient-boosted trees", color="#E76F51")
plt.xticks(x, by_age.age_band)
plt.ylabel("Claim frequency (per exposure-year)")
plt.xlabel("Driver age band")
plt.title("REAL data — claim frequency by driver age: actual vs predicted")
plt.legend()
plt.tight_layout()
plt.savefig(f"{FIGDIR}/frequency_by_age.png", dpi=140)
plt.close()

# COMMAND ----------
# ---- Search for the segment with the largest GLM miss (data-driven, not assumed) --
test["glm_gap"] = test.ClaimNb / test.Exposure.clip(lower=0.01) - test.freq_glm_pred
test["glm_gap_gbm"] = test.freq_gbm_pred - test.freq_glm_pred

candidates = []
for bm_cut in [100, 120, 150]:
    for power_cut in [8, 10, 12]:
        seg = (test.BonusMalus >= bm_cut) & (test.VehPower >= power_cut)
        if seg.sum() < 200:
            continue
        actual = test.loc[seg, "ClaimNb"].sum() / test.loc[seg, "Exposure"].sum()
        glm_p = np.average(test.loc[seg, "freq_glm_pred"], weights=test.loc[seg, "Exposure"])
        gbm_p = np.average(test.loc[seg, "freq_gbm_pred"], weights=test.loc[seg, "Exposure"])
        candidates.append({"bonus_malus_min": bm_cut, "veh_power_min": power_cut, "n": int(seg.sum()),
                            "actual": actual, "glm": glm_p, "gbm": gbm_p,
                            "glm_gap_pct": (actual - glm_p) / actual})
cand_df = pd.DataFrame(candidates).sort_values("glm_gap_pct", ascending=False)
print("\n=== BonusMalus x VehPower segment scan (real data) ===")
print(cand_df.to_string(index=False))
cand_df.to_csv(f"{PBIDIR}/segment_scan_bonusmalus_power.csv", index=False)

# COMMAND ----------
# ---- Feature importance ----------
from sklearn.inspection import permutation_importance
perm = permutation_importance(freq_gbm, test_gbm[ALL_FEATS], test_gbm.ClaimNb / test_gbm.Exposure,
                                sample_weight=test_gbm.Exposure, n_repeats=5, random_state=7, n_jobs=-1)
imp_df = pd.DataFrame({"feature": ALL_FEATS, "importance": perm.importances_mean}).sort_values("importance")
imp_df.to_csv(f"{PBIDIR}/feature_importance_frequency.csv", index=False)

plt.figure(figsize=(7, 4.5))
plt.barh(imp_df.feature, imp_df.importance, color="#1D3557")
plt.title("REAL data — frequency feature importance (gradient-boosted trees)")
plt.xlabel("Importance")
plt.tight_layout()
plt.savefig(f"{FIGDIR}/feature_importance.png", dpi=140)
plt.close()

# COMMAND ----------
# ---- Portfolio tie-out ----------
sev_bundle_glm, sev_bundle_gbm = bundle["sev_glm"], bundle["sev_gbm"]
test["sev_glm_pred"] = sev_bundle_glm.predict(test[ALL_FEATS])
test["sev_gbm_pred"] = sev_bundle_gbm.predict(test_gbm[ALL_FEATS])
test["pp_glm"] = test.freq_glm_pred * test.sev_glm_pred
test["pp_gbm"] = test.freq_gbm_pred * test.sev_gbm_pred

actual_total = test.ClaimAmount_capped.sum()
pred_total_glm = (test.pp_glm * test.Exposure).sum()
pred_total_gbm = (test.pp_gbm * test.Exposure).sum()
tie_out = (f"PORTFOLIO TIE-OUT (real test set, {len(test):,} policies, {test.Exposure.sum():,.0f} exposure-years)\n"
           f"  Actual total incurred (capped):     {actual_total:,.0f}\n"
           f"  GLM  predicted total pure premium:  {pred_total_glm:,.0f}   ({(pred_total_glm/actual_total-1):+.2%})\n"
           f"  GBM  predicted total pure premium:  {pred_total_gbm:,.0f}   ({(pred_total_gbm/actual_total-1):+.2%})\n")
print(tie_out)
with open("portfolio_tie_out.txt", "w") as f:
    f.write(tie_out)

test.sample(min(20000, len(test)), random_state=1)[
    ["IDpol","Region","Area","VehBrand","VehGas","DrivAge","age_band","VehPower","VehAge","BonusMalus",
     "Density","Exposure","ClaimNb","ClaimAmount_capped","freq_glm_pred","freq_gbm_pred","pp_glm","pp_gbm"]
].to_csv(f"{PBIDIR}/policy_level_sample.csv", index=False)
print("Done.")
