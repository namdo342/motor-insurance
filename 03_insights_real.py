# Databricks notebook source
# MAGIC %md
# MAGIC # 03 — Where GLM and the tree model actually diverge (real data)

# COMMAND ----------
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib

FIGDIR = "/home/claude/work/claims-risk-real/figures"
PBIDIR = "/home/claude/work/claims-risk-real/powerbi_exports"
import os
os.makedirs(FIGDIR, exist_ok=True)
os.makedirs(PBIDIR, exist_ok=True)

bundle = joblib.load("/home/claude/work/claims-risk-real/data/models.joblib")
freq_glm, freq_gbm = bundle["freq_glm"], bundle["freq_gbm"]
ALL_FEATS = bundle["ALL_FEATS"]

test = pd.read_csv("/home/claude/work/claims-risk-real/data/test.csv")
test_gbm = pd.read_pickle("/home/claude/work/claims-risk-real/data/test_gbm.pkl")

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
with open("/home/claude/work/claims-risk-real/portfolio_tie_out.txt", "w") as f:
    f.write(tie_out)

test.sample(min(20000, len(test)), random_state=1)[
    ["IDpol","Region","Area","VehBrand","VehGas","DrivAge","age_band","VehPower","VehAge","BonusMalus",
     "Density","Exposure","ClaimNb","ClaimAmount_capped","freq_glm_pred","freq_gbm_pred","pp_glm","pp_gbm"]
].to_csv(f"{PBIDIR}/policy_level_sample.csv", index=False)
print("Done.")
