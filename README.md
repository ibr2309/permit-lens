# permit-lens

Predicts the **declared construction value** of City of Edmonton building permits from what's on the permit
application: job category, work type, building type, floor area, units, location and the free-text job
description. Data comes from the [City of Edmonton Open Data Portal](https://data.edmonton.ca)
("General Building Permits", 2009–2026).

The focus is on an **honest evaluation**: models are tested on years they never saw, ranges are calibrated and
checked for coverage, and the dashboard states the model's limitations.

## Results

Numbers below come from the 40,000-permit sample that ships with the repo (33,407 permits after cleaning).
Training covers 2009–2022, 2023 is used for model selection and interval calibration, and 2024–2026 is the
untouched test set (4,661 permits). They are reproducible with `python -m permit_lens.train`, which writes
[`artifacts/metrics.json`](artifacts/metrics.json).

| Model (test = 2024–2026) | R² (log value) | Median % error | Within ±25% | Within 2× |
|---|---|---|---|---|
| Baseline: median by job category | 0.455 | 40.2% | 32% | 61% |
| Ridge (structured fields) | 0.645 | 28.3% | 46% | 68% |
| Ridge (+ description text) | 0.706 | 36.5% | 35% | 73% |
| XGBoost (structured fields) | 0.673 | 19.3% | 54% | 72% |
| **XGBoost (+ description text)**, selected | **0.704** | **18.6%** | **54%** | **75%** |

What the evaluation shows:

- **A random split overstates accuracy.** The same model scores R² **0.855** on a random 80/20 split but **0.704**
  on future years. The project's earlier R² ≈ 0.84 headline came from a random split.
- **Accuracy depends heavily on the kind of work.** New houses are predicted to within about 5% (median error).
  Home improvements and commercial work miss by about 50–65%.
- **Size and type drive value.** Measured by permutation importance on the test years, floor area matters most,
  followed by job category, units added, the description text and work type. Neighbourhood, zoning and coordinates
  together cost under 0.015 R² when shuffled. Once the size and type of work are known, location adds little
  predictive information. That doesn't mean location has no effect on cost.
- **The ranges are close to calibrated.** The 80% prediction ranges (split-conformal, per job category) contained
  the true value for **75%** of 2024–2026 permits. The small shortfall comes from the data shifting between years.
- **The data changes over time.** Edmonton stopped using the "Combination" job categories after 2023, and recent
  permits omit "units added" more often. The dashboard plots this shift.

## Dashboard

```bash
pip install -r requirements.txt
streamlit run app.py
```

The first launch trains and evaluates every model (about a minute on the sample) and caches the result in
`artifacts/model.joblib`. The dashboard has five pages:

1. **Overview**: value distribution, job-category mix over time, and trends.
2. **Model evaluation**: the out-of-time results, the random-split vs out-of-time comparison, error by category,
   predicted vs actual, and interval coverage.
3. **What drives value**: permutation importance. All the summary text is computed from the model, not hardcoded.
4. **Predict a permit**: an estimate with a calibrated 80% range. Dropdowns only offer combinations that occur in
   the data.
5. **Limitations**.

To use the full dataset, download "General Building Permits" as CSV from data.edmonton.ca, then run:

```bash
python -m permit_lens.train --data General_Building_Permits.csv   # optional: pre-build the model
PERMIT_LENS_DATA=General_Building_Permits.csv streamlit run app.py
```

## How it works

```
permit_lens/
  data.py      cleaning: currency/area parsing, $1k–$50M value range, dimensions parsed from descriptions
  modeling.py  pipelines, temporal split, metrics, conformal intervals, permutation importance
  train.py     CLI: train → evaluate → save bundle + metrics.json
app.py         Streamlit dashboard
tests/         pytest suite (runs in CI)
```

- **Target:** log₁₀ of declared value. Errors are therefore proportional, which suits values that span
  $1k to $50M.
- **Features:**
  - Categoricals are one-hot encoded; levels with fewer than 20 permits are pooled, and unseen levels map to the
    pooled column.
  - Floor area, units, year, month and coordinates are used as numbers. For XGBoost, missing values stay missing
    so the model can learn from missingness.
  - Footprint areas are parsed from text such as "6.10m x 6.10m".
  - The job description is TF-IDF encoded: as a sparse matrix for Ridge, and reduced to 64 SVD components for
    XGBoost.
- **Selection:** the model with the lowest median % error on the validation year (2023) wins. The test years are
  only scored once.
- **Intervals:** Mondrian split-conformal. Per-job-category quantiles of absolute log errors on 2023 give an 80%
  range, and coverage is then measured on 2024–2026.
- **Serving:** the selected model is refit on all years before it is used in the app.

## Limitations

- **The target is declared value, not real cost.** Applicants self-report it; values are often rounded and may
  be understated.
- **Only permits between $1,000 and $50M are modelled.** Values outside that range are treated as placeholders
  or outliers.
- **No forecasting or inflation adjustment.** Tree models don't extrapolate a time trend, and values are in
  nominal dollars.
- **Accuracy drifts as the city's data practices change.** Retrain on recent permits to keep up.
- **Feature importance is not causation.** Correlated inputs share credit.
- **Results are from the sample only.** Numbers on the full ~200k-permit export haven't been re-run with this
  version

## Development

```bash
pip install -r requirements-dev.txt
python -m pytest
```

Stack: Python · pandas · scikit-learn · XGBoost · Plotly · Streamlit
