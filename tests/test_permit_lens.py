import numpy as np
import pandas as pd
import pytest

from permit_lens.data import MIN_VALUE, clean, description_area
from permit_lens.modeling import (
    ALL_FEATURES, ConformalIntervals, GroupMedian, predict_with_interval, regression_metrics,
    temporal_split, train_and_evaluate,
)


def raw_permits(n=1200, years=range(2015, 2023), seed=0):
    """Synthetic export in the raw City of Edmonton format, with a learnable signal."""
    rng = np.random.default_rng(seed)
    jobs = rng.choice(["Home Improvement", "Commercial Final", "Single, Semi-detached & Rowhousing"], n)
    area = rng.uniform(200, 4000, n)
    base = {"Home Improvement": 50, "Commercial Final": 400, "Single, Semi-detached & Rowhousing": 150}
    value = np.array([base[j] for j in jobs]) * area * rng.lognormal(0, 0.2, n)
    year = rng.choice(list(years), n)
    return pd.DataFrame({
        "PERMIT_DATE": [f"{y}-06-01" for y in year], "YEAR": year, "MONTH_NUMBER": rng.integers(1, 13, n),
        "JOB_CATEGORY": jobs, "JOB_DESCRIPTION": [f"To construct a deck ({a / 100:.2f}m x 3.00m)" for a in area],
        "BUILDING_TYPE": "Single Detached House (110)", "WORK_TYPE": rng.choice(["(01) New", None], n),
        "CONSTRUCTION_VALUE": [f"${v:,.0f}" for v in value], "FLOOR_AREA": [f"{a:,.2f}" for a in area],
        "UNITS_ADDED": rng.integers(0, 3, n), "ADDRESS": "1 - MAIN STREET NW",
        "ZONING": rng.choice(["RSF", "RS", None], n), "NEIGHBOURHOOD": rng.choice(["A", "B", "C"], n),
        "LATITUDE": 53.5, "LONGITUDE": -113.5,
    })


def test_description_area_sums_all_dimensions():
    assert description_area("deck (4.88m x 2.00m) and garage 6.10 m X 6.10m") == pytest.approx(4.88 * 2 + 6.1 * 6.1)
    assert np.isnan(description_area("interior alterations"))
    assert np.isnan(description_area(None))


def test_clean_parses_currency_and_filters_placeholders():
    raw = raw_permits(50)
    raw.loc[0, "CONSTRUCTION_VALUE"] = "$0"
    raw.loc[1, "FLOOR_AREA"] = "-5"
    df = clean(raw)
    assert df["value"].min() >= MIN_VALUE
    assert len(df) == 49
    assert (df["floor_area"].dropna() > 0).all()
    assert df["WORK_TYPE"].notna().all() and "Unknown" in set(df["WORK_TYPE"])
    np.testing.assert_allclose(df["log_value"], np.log10(df["value"]))


def test_temporal_split_never_tests_on_training_years():
    df = clean(raw_permits())
    s = temporal_split(df)
    assert s.train["YEAR"].max() < s.valid_year < s.test["YEAR"].min()
    assert len(s.train) + len(s.valid) + len(s.test) == len(df)


def test_metrics_perfect_prediction():
    y = np.log10([1000, 5000, 20000])
    m = regression_metrics(y, y)
    assert m["r2_log"] == pytest.approx(1.0) and m["median_ape"] == 0 and m["within_25pct"] == 100


def test_conformal_intervals_cover_at_requested_level():
    rng = np.random.default_rng(1)
    groups = pd.Series(rng.choice(["a", "b"], 4000))
    noise = np.where(groups == "a", 0.05, 0.5)
    y = rng.normal(0, noise)
    cal = ConformalIntervals.fit(groups[:2000], y[:2000], np.zeros(2000), level=0.8)
    assert cal.by_group["b"] > 5 * cal.by_group["a"]  # wider where errors are larger
    lo, hi = cal.interval(groups[2000:], np.zeros(2000))
    covered = (10 ** y[2000:] >= lo) & (10 ** y[2000:] <= hi)
    assert 0.76 < covered.mean() < 0.84
    assert cal.half_width(pd.Series(["unseen"]))[0] == cal.overall


def test_group_median_baseline_falls_back_for_unseen_groups():
    X = pd.DataFrame({"JOB_CATEGORY": ["a", "a", "b"]})
    m = GroupMedian().fit(X, [1.0, 3.0, 10.0])
    np.testing.assert_allclose(m.predict(pd.DataFrame({"JOB_CATEGORY": ["a", "b", "z"]})), [2.0, 10.0, 3.0])


def test_end_to_end_training_learns_signal():
    df = clean(raw_permits())
    report = train_and_evaluate(df, importance_repeats=1)
    best = report.metrics.loc[report.model_name]
    assert report.metrics["selected"].sum() == 1
    assert best["r2_log"] > 0.7
    assert report.importance.iloc[0]["feature"] in {"floor_area", "JOB_CATEGORY", "desc_area", "JOB_DESCRIPTION"}
    out = predict_with_interval(report, df[ALL_FEATURES].head(5))
    assert (out["lower"] <= out["predicted"]).all() and (out["predicted"] <= out["upper"]).all()
