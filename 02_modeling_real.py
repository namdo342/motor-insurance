# Databricks notebook source
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

df = pd.read_csv("/home/claude/work/claims-risk-real/data/policies_clean.csv")

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
            "/home/claude/work/claims-risk-real/data/models.joblib")
train.to_csv("/home/claude/work/claims-risk-real/data/train.csv", index=False)
test.to_csv("/home/claude/work/claims-risk-real/data/test.csv", index=False)
train_gbm.to_pickle("/home/claude/work/claims-risk-real/data/train_gbm.pkl")
test_gbm.to_pickle("/home/claude/work/claims-risk-real/data/test_gbm.pkl")
pd.DataFrame(freq_results).to_csv("/home/claude/work/claims-risk-real/data/freq_model_comparison.csv", index=False)
pd.DataFrame(sev_results).to_csv("/home/claude/work/claims-risk-real/data/sev_model_comparison.csv", index=False)
print("\nSaved.")
