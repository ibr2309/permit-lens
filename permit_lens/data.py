"""Loading and cleaning the City of Edmonton 'General Building Permits' export."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

RAW_COLUMNS = [
    "PERMIT_DATE", "YEAR", "MONTH_NUMBER", "JOB_CATEGORY", "JOB_DESCRIPTION",
    "BUILDING_TYPE", "WORK_TYPE", "CONSTRUCTION_VALUE", "FLOOR_AREA", "UNITS_ADDED",
    "ADDRESS", "ZONING", "NEIGHBOURHOOD", "LATITUDE", "LONGITUDE",
]

# Declared values outside this range are almost always placeholders ($0, $1) or
# data-entry errors; the model is only claimed to work inside it.
MIN_VALUE = 1_000
MAX_VALUE = 50_000_000

CATEGORICAL = ["JOB_CATEGORY", "WORK_TYPE", "BUILDING_TYPE", "ZONING", "NEIGHBOURHOOD"]

# "6.10m x 6.10m", "4.88 m X 3.96m"
_DIM_RE = re.compile(r"(\d+(?:\.\d+)?)\s*m\s*[x×]\s*(\d+(?:\.\d+)?)\s*m", re.IGNORECASE)


def _to_number(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s.astype(str).str.replace(r"[$,\s]", "", regex=True), errors="coerce")


def description_area(text: str) -> float:
    """Sum of every 'A m x B m' footprint mentioned in a job description, in m²."""
    if not isinstance(text, str):
        return np.nan
    areas = [float(a) * float(b) for a, b in _DIM_RE.findall(text)]
    return float(sum(areas)) if areas else np.nan


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Turn the raw export into one modelling-ready row per permit."""
    df = df[[c for c in RAW_COLUMNS if c in df.columns]].copy()

    df["value"] = _to_number(df["CONSTRUCTION_VALUE"])
    df = df[df["value"].between(MIN_VALUE, MAX_VALUE)].copy()

    df["PERMIT_DATE"] = pd.to_datetime(df["PERMIT_DATE"], errors="coerce")
    df["YEAR"] = pd.to_numeric(df["YEAR"], errors="coerce")
    df["YEAR"] = df["YEAR"].fillna(df["PERMIT_DATE"].dt.year)
    df["MONTH_NUMBER"] = pd.to_numeric(df["MONTH_NUMBER"], errors="coerce")
    df["MONTH_NUMBER"] = df["MONTH_NUMBER"].fillna(df["PERMIT_DATE"].dt.month)
    df = df.dropna(subset=["YEAR", "MONTH_NUMBER"])
    df["YEAR"] = df["YEAR"].astype(int)
    df["MONTH_NUMBER"] = df["MONTH_NUMBER"].astype(int)

    fa = _to_number(df["FLOOR_AREA"])
    df["floor_area"] = fa.where(fa > 0)
    units = pd.to_numeric(df["UNITS_ADDED"], errors="coerce")
    df["units_added"] = units
    df["latitude"] = pd.to_numeric(df["LATITUDE"], errors="coerce")
    df["longitude"] = pd.to_numeric(df["LONGITUDE"], errors="coerce")

    df["JOB_DESCRIPTION"] = df["JOB_DESCRIPTION"].fillna("").astype(str)
    df["desc_area"] = df["JOB_DESCRIPTION"].map(description_area)
    for c in CATEGORICAL:
        df[c] = df[c].fillna("Unknown").astype(str).str.strip()

    df["log_value"] = np.log10(df["value"])
    df = df.drop(columns=["CONSTRUCTION_VALUE", "FLOOR_AREA", "UNITS_ADDED", "LATITUDE", "LONGITUDE"])
    df = df.drop_duplicates()
    return df.reset_index(drop=True)


def load(path: str | Path) -> pd.DataFrame:
    raw = pd.read_csv(path, usecols=lambda c: c in RAW_COLUMNS, low_memory=False)
    return clean(raw)
