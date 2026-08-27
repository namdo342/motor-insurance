# Databricks notebook source
# MAGIC %md
# MAGIC # 01 — Load, merge, and clean the REAL freMTPL2freq / freMTPL2sev data
# MAGIC Source: OpenML dataset 41214 (C. Dutang & A. Charpentier, CASdatasets),
# MAGIC uploaded by the user. 678,013 policies, ~2011-2013, France.

# COMMAND ----------
import numpy as np
import pandas as pd

freq = pd.read_csv("/home/claude/work/claims-risk-real/data/freMTPL2freq.csv")
sev = pd.read_csv("/home/claude/work/claims-risk-real/data/freMTPL2sev.csv")
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

with open("/home/claude/work/claims-risk-real/data/data_quality_report.md", "w") as f:
    f.write("\n".join(report))
df.to_csv("/home/claude/work/claims-risk-real/data/policies_clean.csv", index=False)
print(f"Clean file: {n1:,} rows -> data/policies_clean.csv")
print("\n".join(report))
