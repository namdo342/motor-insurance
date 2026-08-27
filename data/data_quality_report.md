# Data quality report — REAL freMTPL2freq / freMTPL2sev

Raw policies: **678,013**  
Raw severity records: **26,639**

## 1. Orphaned claims
**195** severity records reference an IDpol not present in the policy table. Dropped (no risk factors to model them against).

## 2. ClaimNb vs. actual linked severity records
**9,117** policies (**1.3%** of the book) have a `ClaimNb` that does not match the number of severity records actually linked to them, almost entirely `ClaimNb > 0` with zero linked severity rows (9,116 policies), 97% of which have `IDpol <= 24500`. This matches a data-linkage issue in this exact dataset that is documented in the actuarial literature (Wuthrich et al.), not a mistake in this pipeline.

**Decision:** keep `ClaimNb` as-is for the FREQUENCY model (it is still the field the source labels as the claim count). For the SEVERITY model, only policies with a genuine linked claim amount are usable, the 9,116 policies with no linked amount are excluded from severity training only, flagged via `severity_usable`.

## 3. Exposure > 1.0
**1,224** policies (0.18%), max observed 2.01. Capped at 1.0 rather than dropped, these are almost certainly rounding/measurement artefacts on otherwise-usable rows, not corrupted records.

Exposure <= 0: **0** rows, dropped.

## 4. Extreme VehAge / DrivAge
VehAge > 25: **2,619** rows (max 100), capped at 25. DrivAge > 90: **401** rows (max 100), capped at 90. Capped, not dropped, the policy is still valid, only the extreme tail of one feature is smoothed.

## 5. Severity outliers
**125** claims above the 99.5th percentile (> 35,630), including one single claim of 4,075,401. Kept in the portfolio total (`ClaimAmount`), capped in `ClaimAmount_capped` for model fitting.


## Summary
Policies in: **678,013** -> out: **678,013** (0 dropped, 0.00%). Total claims linked: **26,444** of 36,102 reported in ClaimNb.
