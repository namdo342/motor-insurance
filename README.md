# Motor Insurance Risk Analysis

## Why I built this

I interned at an insurance company over the summer, reconciling claims and checking the calculations behind how a policy gets priced. Those prices come from a GLM, a fairly old but well-trusted statistical model. I wanted to know what a more flexible model would actually say about the same risk, so I got the real dataset this kind of pricing research is usually done on and built one myself.

## The data

678,013 real French motor insurance policies from around 2011-2013 (OpenML dataset 41214, originally from Dutang & Charpentier's *CASdatasets*). Not included in this repo directly, the files are 37-48MB each, download from [Hugging Face](https://huggingface.co/datasets/mabilton/fremtpl2) (easiest, plain CSV) or the [original OpenML page](https://www.openml.org/d/41214), then drop both CSVs into `data/`.

## What I actually did

The first step was just cleaning the thing, and that turned out to be more interesting than I expected. Merging the two source tables, policies and claims, surfaced a real defect that's actually documented in the literature on this exact dataset: 9,116 policies show a claim on file but have no matching payment record anywhere. I kept those policies for frequency modelling, the claim count itself is still valid, but excluded them from the severity model, since there's no amount to train on there.

Once the data was clean, I built a Poisson GLM for claim frequency and a Gamma GLM for claim severity, the standard actuarial approach, then a gradient-boosted-tree version of each to compare against. Same target, same held-out test set, just a different way of representing how the risk factors relate to the outcome.

## What I found

Averaged across the whole book, the two models barely disagree. The tree beats the GLM by about 5.5% on frequency deviance, a gap small enough that you could reasonably conclude the extra complexity isn't worth it.

Except that's the wrong conclusion. Broken down by driver age, the GLM underprices the youngest drivers, 18 to 22, by 26%, a gap that's completely invisible in the aggregate number. Two errors in opposite directions can cancel out on average and still cost real money on the one segment where it actually matters.

I also want to flag something that didn't work. An earlier version of this, run on simulated data, was built around the idea that young drivers in high-powered cars specifically would show the biggest gap. That interaction didn't clearly show up once I moved to the real data. I'm reporting that as a negative result rather than reshaping the analysis until it matched the story I expected going in.

One more thing worth mentioning: the portfolio total looked off by 37-42% at first, which would normally point to a real problem with the models. It wasn't the models. Only 72.6% of the claims in the data have a linked payment amount, the same defect mentioned above, so the "actual" total is itself an undercount. Correct for that and both models land within a few percent of the true total.

## Running it

```bash
pip install pandas numpy scikit-learn matplotlib joblib
python3 motor_insurance_risk_analysis.py
```

Runs in under a minute on the full dataset. It's written as a single file with Databricks-style cell markers (`# COMMAND ----------`), so it imports directly as a notebook too if that's more useful.

## Power BI

I haven't built the actual `.pbix` file yet. `powerbi_exports/` has the cleaned, aggregated CSVs ready to go: `frequency_by_age_band.csv`, `feature_importance_frequency.csv`, `segment_scan_bonusmalus_power.csv`, and a 20,000-row `policy_level_sample.csv` for drill-through.

A line chart of frequency by age band recreates the main finding directly. The segment scan file is worth a table with conditional formatting on `glm_gap_pct`, so where the GLM misses jumps out at a glance.

## What's in the repo

`motor_insurance_risk_analysis.py` is the whole pipeline. `data/data_quality_report.md` has every cleaning decision with counts. `figures/` has the two charts referenced above, `portfolio_tie_out.txt` has the reconciliation numbers, and `powerbi_exports/` has the CSVs.
