"""Feature pipelines, models, out-of-time evaluation and calibrated intervals."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from xgboost import XGBRegressor

from .data import CATEGORICAL

TARGET = "log_value"
TEXT = "JOB_DESCRIPTION"
LOG_NUMERIC = ["floor_area", "desc_area"]
NUMERIC = ["YEAR", "MONTH_NUMBER", "units_added", "latitude", "longitude"]
STRUCTURED = CATEGORICAL + LOG_NUMERIC + NUMERIC
ALL_FEATURES = STRUCTURED + [TEXT]

FEATURE_LABELS = {
    "JOB_CATEGORY": "Job category", "WORK_TYPE": "Work type", "BUILDING_TYPE": "Building type",
    "ZONING": "Zoning", "NEIGHBOURHOOD": "Neighbourhood", "floor_area": "Floor area",
    "desc_area": "Dimensions in description", "YEAR": "Year", "MONTH_NUMBER": "Month",
    "units_added": "Units added", "latitude": "Latitude", "longitude": "Longitude",
    TEXT: "Job description text",
}

INTERVAL_LEVEL = 0.80
MIN_GROUP_SIZE = 50
SEED = 42


# ── Pipelines ─────────────────────────────────────────────────────────────────

def _onehot() -> OneHotEncoder:
    # Rare levels (<20 permits) share one 'infrequent' column; unseen levels map there too.
    return OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=20)


def _log1p(x):
    return np.log1p(np.asarray(x, dtype=float))


def _log_tf() -> FunctionTransformer:
    return FunctionTransformer(_log1p, feature_names_out="one-to-one")


def _tfidf() -> TfidfVectorizer:
    return TfidfVectorizer(ngram_range=(1, 2), min_df=5, max_features=20_000,
                           sublinear_tf=True, strip_accents="unicode")


def ridge_pipeline(text: bool) -> Pipeline:
    """Linear baseline: one-hot categoricals, imputed + standardized numerics, sparse TF-IDF."""
    num = make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler())
    parts = [
        ("cat", _onehot(), CATEGORICAL),
        ("lognum", make_pipeline(_log_tf(), num), LOG_NUMERIC),
        ("num", clone(num), NUMERIC),
    ]
    if text:
        parts.append(("text", _tfidf(), TEXT))
    return Pipeline([("features", ColumnTransformer(parts)), ("model", Ridge(alpha=3.0))])


def xgb_pipeline(text: bool) -> Pipeline:
    """Gradient-boosted trees on dense features; NaN is kept so XGBoost learns missingness."""
    parts = [
        ("cat", _onehot(), CATEGORICAL),
        ("lognum", _log_tf(), LOG_NUMERIC),
        ("num", "passthrough", NUMERIC),
    ]
    if text:
        parts.append(("text", make_pipeline(_tfidf(), TruncatedSVD(64, random_state=SEED)), TEXT))
    features = ColumnTransformer(parts, sparse_threshold=0.0)
    model = XGBRegressor(n_estimators=900, learning_rate=0.04, max_depth=7, min_child_weight=3,
                         subsample=0.8, colsample_bytree=0.7, reg_lambda=1.0,
                         tree_method="hist", random_state=SEED, n_jobs=-1, verbosity=0)
    return Pipeline([("features", features), ("model", model)])


class GroupMedian(BaseEstimator, RegressorMixin):
    """Naive baseline: the training median of log value for the permit's job category."""

    def __init__(self, column: str = "JOB_CATEGORY"):
        self.column = column

    def fit(self, X, y):
        y = pd.Series(np.asarray(y), index=X.index)
        self.medians_ = y.groupby(X[self.column]).median().to_dict()
        self.global_ = float(y.median())
        return self

    def predict(self, X):
        return X[self.column].map(self.medians_).fillna(self.global_).to_numpy(dtype=float)


def candidate_models() -> dict[str, BaseEstimator]:
    return {
        "Baseline: median by job category": GroupMedian(),
        "Ridge (structured)": ridge_pipeline(text=False),
        "Ridge (+ description text)": ridge_pipeline(text=True),
        "XGBoost (structured)": xgb_pipeline(text=False),
        "XGBoost (+ description text)": xgb_pipeline(text=True),
    }


# ── Splits & metrics ──────────────────────────────────────────────────────────

@dataclass
class TemporalSplit:
    train: pd.DataFrame
    valid: pd.DataFrame
    test: pd.DataFrame
    train_years: tuple[int, int]
    valid_year: int
    test_years: tuple[int, int]


def temporal_split(df: pd.DataFrame, n_test_years: int = 3) -> TemporalSplit:
    """Train on the past, select on the next year, test on the most recent years.

    This mirrors real use: the model is always predicting permits from years it has never seen.
    """
    years = sorted(df["YEAR"].unique())
    if len(years) < n_test_years + 2:
        raise ValueError(f"Need at least {n_test_years + 2} distinct years, got {len(years)}")
    test_start = years[-n_test_years]
    valid_year = years[-n_test_years - 1]
    train = df[df["YEAR"] < valid_year]
    valid = df[df["YEAR"] == valid_year]
    test = df[df["YEAR"] >= test_start]
    return TemporalSplit(train, valid, test, (int(train["YEAR"].min()), int(train["YEAR"].max())),
                         int(valid_year), (int(test_start), int(years[-1])))


def regression_metrics(y_log: np.ndarray, p_log: np.ndarray) -> dict[str, float]:
    y, p = 10 ** np.asarray(y_log), 10 ** np.asarray(p_log)
    ape = np.abs(p - y) / y
    return {
        "r2_log": float(r2_score(y_log, p_log)),
        "median_ape": float(np.median(ape) * 100),
        "within_25pct": float(np.mean(ape <= 0.25) * 100),
        "within_2x": float(np.mean(np.abs(np.asarray(p_log) - np.asarray(y_log)) <= np.log10(2)) * 100),
        "mae_dollars": float(mean_absolute_error(y, p)),
    }


# ── Calibrated intervals (Mondrian split-conformal on log residuals) ──────────

@dataclass
class ConformalIntervals:
    """Per-job-category residual quantiles giving ~INTERVAL_LEVEL coverage in log10 space."""

    level: float
    by_group: dict[str, float]
    overall: float
    column: str = "JOB_CATEGORY"

    @staticmethod
    def _quantile(abs_resid: np.ndarray, level: float) -> float:
        n = len(abs_resid)
        q = min(1.0, np.ceil((n + 1) * level) / n)
        return float(np.quantile(abs_resid, q, method="higher"))

    @classmethod
    def fit(cls, groups: pd.Series, y_log, p_log, level: float = INTERVAL_LEVEL) -> "ConformalIntervals":
        resid = pd.Series(np.abs(np.asarray(y_log) - np.asarray(p_log)), index=groups.index)
        by_group = {
            g: cls._quantile(r.to_numpy(), level)
            for g, r in resid.groupby(groups) if len(r) >= MIN_GROUP_SIZE
        }
        return cls(level, by_group, cls._quantile(resid.to_numpy(), level))

    def half_width(self, groups: pd.Series) -> np.ndarray:
        return groups.map(self.by_group).fillna(self.overall).to_numpy(dtype=float)

    def interval(self, groups: pd.Series, p_log) -> tuple[np.ndarray, np.ndarray]:
        w = self.half_width(groups)
        p_log = np.asarray(p_log)
        return 10 ** (p_log - w), 10 ** (p_log + w)


# ── End-to-end training ───────────────────────────────────────────────────────

@dataclass
class TrainingReport:
    model: BaseEstimator
    model_name: str
    split: dict
    metrics: pd.DataFrame
    random_split_metrics: dict[str, float]
    intervals: ConformalIntervals
    interval_coverage: dict[str, float]
    importance: pd.DataFrame
    test_predictions: pd.DataFrame
    meta: dict = field(default_factory=dict)


def _fit_predict(model, train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    model.fit(train[ALL_FEATURES], train[TARGET])
    return model.predict(test[ALL_FEATURES])


def train_and_evaluate(df: pd.DataFrame, importance_repeats: int = 3, verbose: bool = False) -> TrainingReport:
    log = print if verbose else (lambda *a, **k: None)
    split = temporal_split(df)
    fit_df = pd.concat([split.train, split.valid])

    # 1. Model selection on the validation year (never on the test years).
    candidates = candidate_models()
    rows, valid_scores = [], {}
    for name, est in candidates.items():
        log(f"  fitting {name}")
        p_valid = _fit_predict(clone(est), split.train, split.valid)
        valid_scores[name] = regression_metrics(split.valid[TARGET], p_valid)["median_ape"]
        fitted = clone(est).fit(fit_df[ALL_FEATURES], fit_df[TARGET])
        p_test = fitted.predict(split.test[ALL_FEATURES])
        rows.append({"model": name, **regression_metrics(split.test[TARGET], p_test)})
        candidates[name] = (fitted, p_test)
    best_name = min(valid_scores, key=valid_scores.get)
    best_model, best_test_pred = candidates[best_name]
    metrics = pd.DataFrame(rows).set_index("model")
    metrics["selected"] = metrics.index == best_name

    # 2. Intervals: calibrate on validation-year residuals of a model trained before it,
    #    then measure actual coverage on the untouched test years.
    log("  calibrating intervals")
    p_valid = _fit_predict(clone(candidate_models()[best_name]), split.train, split.valid)
    cal = ConformalIntervals.fit(split.valid["JOB_CATEGORY"], split.valid[TARGET], p_valid)
    w_test = cal.half_width(split.test["JOB_CATEGORY"])
    abs_err = np.abs(split.test[TARGET].to_numpy() - best_test_pred)
    coverage = {"level": INTERVAL_LEVEL * 100, "test_coverage": float(np.mean(abs_err <= w_test) * 100),
                "median_width_factor": float(np.median(10 ** w_test))}

    # 3. What a random split would have reported, to show how optimistic it is.
    log("  random-split comparison")
    tr, te = train_test_split(df, test_size=0.2, random_state=SEED)
    random_metrics = regression_metrics(te[TARGET], _fit_predict(clone(candidate_models()[best_name]), tr, te))

    # 4. Permutation importance on the test years, one raw input column at a time.
    log("  permutation importance")
    imp = permutation_importance(best_model, split.test[ALL_FEATURES], split.test[TARGET],
                                 n_repeats=importance_repeats, random_state=SEED, scoring="r2", n_jobs=1)
    importance = (pd.DataFrame({"feature": ALL_FEATURES, "r2_drop": imp.importances_mean,
                                "r2_drop_std": imp.importances_std})
                  .assign(label=lambda d: d["feature"].map(FEATURE_LABELS))
                  .sort_values("r2_drop", ascending=False).reset_index(drop=True))

    test_predictions = split.test[["YEAR", "JOB_CATEGORY", "WORK_TYPE", "value"]].copy()
    test_predictions["predicted"] = 10 ** best_test_pred
    lo, hi = cal.interval(split.test["JOB_CATEGORY"], best_test_pred)
    test_predictions["lower"], test_predictions["upper"] = lo, hi

    # 5. The model served by the app is refit on every year available.
    log("  refitting on all data")
    final = clone(candidate_models()[best_name]).fit(df[ALL_FEATURES], df[TARGET])

    meta = {
        "n_permits": int(len(df)),
        "year_min": int(df["YEAR"].min()), "year_max": int(df["YEAR"].max()),
        "categories": {c: sorted(df[c].unique().tolist()) for c in CATEGORICAL},
        "category_counts": {c: df[c].value_counts().to_dict() for c in CATEGORICAL},
        "validation_median_ape": valid_scores,
    }
    return TrainingReport(
        model=final, model_name=best_name,
        split={"train_years": split.train_years, "valid_year": split.valid_year,
               "test_years": split.test_years, "n_train": len(split.train),
               "n_valid": len(split.valid), "n_test": len(split.test)},
        metrics=metrics, random_split_metrics=random_metrics, intervals=cal,
        interval_coverage=coverage, importance=importance,
        test_predictions=test_predictions, meta=meta,
    )


def predict_with_interval(report: TrainingReport, X: pd.DataFrame) -> pd.DataFrame:
    p_log = report.model.predict(X[ALL_FEATURES])
    lo, hi = report.intervals.interval(X["JOB_CATEGORY"], p_log)
    return pd.DataFrame({"predicted": 10 ** p_log, "lower": lo, "upper": hi}, index=X.index)
