# Edmonton Building Permit Construction Value Predictor

A machine learning project that predicts the declared **construction value** of a City of Edmonton building permit from its characteristics — trained on **204,590 real permit records (2009–2026)** from the [City of Edmonton Open Data Portal](https://data.edmonton.ca).

## What it does

Given a permit's job category, building type, work type, floor area, neighbourhood, and zoning, the model estimates its construction value. The repo includes a full EDA → feature engineering → model comparison → interactive prediction pipeline, served as a 4-page Streamlit dashboard.

## Data

- **Source:** City of Edmonton Open Data Portal — "General Building Permits" (real, public, verifiable)
- **Size:** 242,479 raw permits → 204,590 after filtering to valid construction values ($500–$50M)
- **Total declared value:** ~$57.6B across all permits
- A 40,000-row **sample** ships with the repo (`General_Building_Permits_sample.csv`, ~17MB) so the app runs out of the box. For the full headline numbers below, download the complete CSV from the portal and point `DATA_PATH` in `app.py` at it.

## Results (full 204k dataset)

Target is log₁₀(construction value); 80/20 train-test split, 40,918 permits held out.

| Model | R² (log) | Mean abs error | Median abs % error |
|-------|----------|----------------|--------------------|
| Linear Regression | 0.557 | $204,987 | 63.9% |
| Ridge Regression | 0.557 | $204,987 | 63.9% |
| Random Forest | 0.839 | $88,652 | **16.9%** |
| Gradient Boosting | 0.838 | $97,402 | 19.9% |
| **XGBoost** | **0.842** | $98,479 | 18.8% |

Tree ensembles dramatically outperform linear models (R²=0.84 vs 0.56) because construction value depends on non-linear interactions between project type, building type, and size that linear regression can't capture.

## Feature importance (XGBoost)

| Feature | Importance |
|---------|-----------|
| Job Category | 56.7% |
| Floor Area (log) | 27.6% |
| Work Type | 11.3% |
| Building Type | 1.8% |
| Everything else (incl. location) | <2% |

**Notable modeling decision:** `UNITS_ADDED` initially dominated importance at ~75%, but the model scored *identically* without it (R²=0.842 either way). It was deliberately dropped to keep the feature set simpler and more interpretable — a tradeoff worth defending in an interview. Equally notable: neighbourhood and zoning together contribute <1%, meaning value is driven by *what* is built, not *where*.

## Run it

```bash
pip install -r requirements.txt
streamlit run app.py
```

The app loads the data, trains all 5 models (cached), and serves four pages:

1. **Overview & EDA** — KPIs, value distribution, permits/year, value by category, top neighbourhoods
2. **Model Comparison** — metrics table, R² and error charts, predicted-vs-actual scatter
3. **Feature Analysis** — importance chart, key findings, value-by-work-type box plots
4. **Value Predictor** — enter permit characteristics, get a value estimate with a confidence band

## Stack

Python · pandas · scikit-learn · XGBoost · Plotly · Streamlit

## Resume bullet

> Built an ML pipeline on 204,000+ real City of Edmonton building permit records to predict construction value; engineered 9 features, compared 5 regression models, and achieved R²=0.84 (XGBoost) with ~17% median error; shipped an interactive 4-page Streamlit dashboard with live prediction. Surfaced that project *type* and *size* — not location — drive value, and dropped a dominant-but-redundant feature after validating no performance loss.
