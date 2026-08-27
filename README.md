# Motor Insurance Risk Analysis — GLM vs. Gradient-Boosted Trees

A frequency-severity claims-pricing pipeline (the standard actuarial pricing
setup) — Poisson GLM for claim frequency, Gamma GLM for claim severity —
benchmarked against a gradient-boosted-tree challenger, on the real French
Motor Third-Party Liability dataset (678,013 policies, ~2011-2013).

## Data

Not included in this repo (37-48MB per file — see `.gitignore`). Download from
either source, then drop both CSVs into `data/`:

- **Easiest (plain CSV):** https://huggingface.co/datasets/mabilton/fremtpl2
- **Original source:** https://www.openml.org/d/41214 (OpenML dataset 41214;
  C. Dutang & A. Charpentier, *CASdatasets*)

Expected files: `data/freMTPL2freq.csv` (678,013 rows, policy risk factors)
and `data/freMTPL2sev.csv` (26,639 rows, claim amounts, linked by `IDpol`).

## Run

```bash
pip install pandas numpy scikit-learn matplotlib joblib
python3 motor_insurance_risk_analysis.py
```

Single file, structured as Databricks notebook cells (`# COMMAND ----------`)
so it also imports directly as a Databricks notebook. Runs in under a minute
on the full 678k-row dataset.

## What it does

1. **Cleans and merges** `freMTPL2freq` + `freMTPL2sev`, including surfacing
   a published data-linkage defect in this exact dataset (9,116 policies with
   a claim count but no matching severity record), documented in
   `data/data_quality_report.md` after running.
2. **Fits a Poisson GLM** (frequency, exposure-weighted) and a **Gamma GLM**
   (severity), the actuarial-industry-standard approach.
3. **Fits a gradient-boosted-tree challenger** (`HistGradientBoostingRegressor`,
   native Poisson/Gamma loss, the closest sklearn equivalent to XGBoost's
   `count:poisson` / `reg:gamma` objectives) on the same targets.
4. **Evaluates both on a held-out test set** via Poisson/Gamma deviance, then
   drills into segments (driver age, BonusMalus x VehPower) to find where the
   two disagree, not just whether they do on average.
5. **Reconciles the portfolio total** against actual incurred losses as a
   sanity check, and explains an apparent 37-42% "miss" that turns out to be
   a data-coverage artefact, not a model failure.

## Headline result

The gradient-boosted model beats the GLM by ~5.5% on frequency deviance
overall. Broken down by driver age, the GLM underprices the youngest drivers
(18-22) by **26%**, a gap invisible in the aggregate metric. Also includes an
honest negative result: a hypothesised age x vehicle-power interaction from
an earlier synthetic-data pass did not clearly replicate on the real data,
reported as such rather than reshaped until it did.

## Power BI

No `.pbix` is included (this sandbox has no Power BI Desktop to build one in).
`powerbi_exports/` has clean, import-ready CSVs instead:

- `frequency_by_age_band.csv` — actual vs. GLM vs. GBM frequency by age band
- `segment_scan_bonusmalus_power.csv` — the BonusMalus x VehPower scan (the honest negative result)
- `feature_importance_frequency.csv` — permutation importance for the tree model
- `policy_level_sample.csv` — 20,000-row policy-level sample for drill-through

**Suggested build:** a line chart of actual/GLM/GBM frequency by `age_band`
(recreates `figures/frequency_by_age.png` natively); a table on
`segment_scan_bonusmalus_power` with conditional formatting on `glm_gap_pct`
to surface where the GLM misses at a glance; a bar chart of
`feature_importance_frequency`.

## Output

- `data/data_quality_report.md` — every cleaning decision, with counts
- `figures/frequency_by_age.png`, `figures/feature_importance.png`
- `powerbi_exports/*.csv` — aggregated tables for Power BI / further analysis
- `portfolio_tie_out.txt`
