"""Train, evaluate and save a model bundle.

    python -m permit_lens.train --data General_Building_Permits_sample.csv --out artifacts/model.joblib
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import joblib

from . import __version__
from .data import load
from .modeling import TrainingReport, train_and_evaluate


def save(report: TrainingReport, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"version": __version__, "report": report}, path, compress=3)


def load_report(path: str | Path) -> TrainingReport:
    bundle = joblib.load(path)
    if bundle.get("version") != __version__:
        raise ValueError(f"Bundle built with permit-lens {bundle.get('version')}, expected {__version__}")
    return bundle["report"]


def summary(report: TrainingReport) -> dict:
    return {
        "selected_model": report.model_name,
        "split": report.split,
        "test_metrics": report.metrics.round(3).to_dict(orient="index"),
        "random_split_metrics": {k: round(v, 3) for k, v in report.random_split_metrics.items()},
        "interval_coverage": {k: round(v, 2) for k, v in report.interval_coverage.items()},
        "importance": report.importance[["label", "r2_drop"]].round(4).to_dict(orient="records"),
    }


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="General_Building_Permits_sample.csv")
    ap.add_argument("--out", default="artifacts/model.joblib")
    ap.add_argument("--metrics", default="artifacts/metrics.json")
    args = ap.parse_args(argv)

    t0 = time.time()
    df = load(args.data)
    print(f"Loaded {len(df):,} permits from {args.data}")
    report = train_and_evaluate(df, verbose=True)
    save(report, args.out)
    Path(args.metrics).write_text(json.dumps(summary(report), indent=2, default=str))
    print(json.dumps(summary(report), indent=2, default=str))
    print(f"Saved {args.out} and {args.metrics} in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
