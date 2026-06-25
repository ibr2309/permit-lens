# permit-lens

Predicts the construction value of City of Edmonton building permits using real open data. Built to practice end-to-end ML on a messy, real-world dataset.

## Results

- **204,590 real permits** (2009–2026) from the [City of Edmonton Open Data Portal]
- **XGBoost R²=0.84**, ~17% median error on held-out test set
- Tree ensembles (R²=0.84) outperform linear regression (R²=0.56) by a wide margin
- Surprising finding: **location barely matters** — neighbourhood and zoning contribute <2% of feature importance. What gets built, not where, drives value.

## Stack

Python · pandas · scikit-learn · XGBoost · Plotly · Streamlit

## Run it

```bash
pip install -r requirements.txt
streamlit run app.py
```

A 40k-row sample ships with the repo so it runs immediately. Download the full CSV from data.edmonton.ca and update `DATA_PATH` in `app.py` for the full dataset numbers.

## Dashboard

Four pages: EDA, model comparison, feature analysis, and a live value predictor.
