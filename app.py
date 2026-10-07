import os
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from permit_lens.data import CATEGORICAL, MAX_VALUE, MIN_VALUE, description_area, load
from permit_lens.modeling import ALL_FEATURES, predict_with_interval, train_and_evaluate
from permit_lens.train import load_report, save

DATA_PATH = os.environ.get("PERMIT_LENS_DATA", "General_Building_Permits_sample.csv")
MODEL_PATH = Path(os.environ.get("PERMIT_LENS_MODEL", "artifacts/model.joblib"))

st.set_page_config(page_title="permit-lens · Edmonton permit values", page_icon="🏗️", layout="wide")

BLUE, GREEN, AMBER, RED, GREY = "#3b82f6", "#10b981", "#f59e0b", "#ef4444", "#94a3b8"


def style(fig, height=340):
    fig.update_layout(template="plotly_dark", height=height, margin=dict(l=0, r=0, t=10, b=0),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    return fig


def money(x):
    return f"${x:,.0f}"


@st.cache_data(show_spinner="Loading and cleaning permits…")
def get_data(path):
    return load(path)


@st.cache_resource(show_spinner="Training and evaluating models (about a minute on the sample)…")
def get_report(path, n_rows):
    if MODEL_PATH.exists():
        try:
            report = load_report(MODEL_PATH)
            if report.meta.get("n_permits") == n_rows:
                return report
        except Exception:
            pass
    report = train_and_evaluate(get_data(path))
    try:
        save(report, MODEL_PATH)
    except OSError:
        pass
    return report


try:
    df = get_data(DATA_PATH)
except FileNotFoundError:
    st.error(f"Could not find `{DATA_PATH}`. Download 'General Building Permits' as CSV from "
             "data.edmonton.ca and set the PERMIT_LENS_DATA environment variable to its path.")
    st.stop()

report = get_report(DATA_PATH, len(df))
metrics = report.metrics
best = metrics.loc[report.model_name]
sp = report.split
test_label = f"{sp['test_years'][0]}–{sp['test_years'][1]}"

with st.sidebar:
    st.markdown("## 🏗️ permit-lens")
    st.caption("Construction value of City of Edmonton building permits")
    st.markdown(f"**{len(df):,}** permits · {df['YEAR'].min()}–{df['YEAR'].max()}")
    st.markdown(f"Model: **{report.model_name}**")
    st.markdown(f"Out-of-time R² **{best['r2_log']:.2f}** · median error **{best['median_ape']:.0f}%**")
    st.markdown("---")
    page = st.radio("Page", ["Overview", "Model evaluation", "What drives value", "Predict a permit", "Limitations"])
    st.markdown("---")
    st.caption(f"Data: data.edmonton.ca (`{Path(DATA_PATH).name}`)")

# ══ Overview ═════════════════════════════════════════════════════════════════
if page == "Overview":
    st.title("Edmonton building permits")
    st.caption(f"Declared construction value, permits between {money(MIN_VALUE)} and {money(MAX_VALUE)}")
    c = st.columns(4)
    c[0].metric("Permits", f"{len(df):,}")
    c[1].metric("Total declared value", f"${df['value'].sum() / 1e9:.1f}B")
    c[2].metric("Median permit", money(df["value"].median()))
    c[3].metric("Mean permit", money(df["value"].mean()), help="Pulled far above the median by a few large projects.")

    l, r = st.columns(2)
    with l:
        st.subheader("Value distribution")
        fig = px.histogram(df, x="value", nbins=70, log_x=True, color_discrete_sequence=[BLUE])
        st.plotly_chart(style(fig.update_xaxes(title="Declared value ($, log scale)")), width="stretch")
    with r:
        st.subheader("Median value by job category")
        vc = df.groupby("JOB_CATEGORY")["value"].agg(["median", "size"]).query("size >= 30").sort_values("median")
        fig = px.bar(vc.reset_index(), x="median", y="JOB_CATEGORY", orientation="h", color_discrete_sequence=[BLUE],
                     hover_data={"size": True})
        st.plotly_chart(style(fig.update_layout(xaxis_title="Median value ($)", yaxis_title="")), width="stretch")

    st.subheader("The job-category mix changes over time")
    st.caption("Edmonton's permit system stopped issuing the 'Combination' categories after 2023, so recent permits "
               "look different from the history the model learns from. This is why evaluation is done out-of-time.")
    mix = df.groupby(["YEAR", "JOB_CATEGORY"]).size().reset_index(name="n")
    mix["share"] = mix["n"] / mix.groupby("YEAR")["n"].transform("sum")
    fig = px.area(mix, x="YEAR", y="share", color="JOB_CATEGORY")
    st.plotly_chart(style(fig.update_layout(yaxis_tickformat=".0%", legend_title=""), 380), width="stretch")

    l, r = st.columns(2)
    with l:
        st.subheader("Permits per year")
        fig = px.bar(df.groupby("YEAR").size().reset_index(name="permits"), x="YEAR", y="permits",
                     color_discrete_sequence=[GREEN])
        st.plotly_chart(style(fig), width="stretch")
    with r:
        st.subheader("Median value per year (nominal $)")
        fig = px.line(df.groupby("YEAR")["value"].median().reset_index(), x="YEAR", y="value", markers=True,
                      color_discrete_sequence=[AMBER])
        st.plotly_chart(style(fig.update_yaxes(title="Median value ($)")), width="stretch")

# ══ Model evaluation ═════════════════════════════════════════════════════════
elif page == "Model evaluation":
    st.title("Model evaluation")
    st.markdown(
        f"Models train on **{sp['train_years'][0]}–{sp['train_years'][1]}** ({sp['n_train']:,} permits), are compared on "
        f"**{sp['valid_year']}** ({sp['n_valid']:,}) to pick a winner, and are scored once on **{test_label}** "
        f"({sp['n_test']:,}) — years no model saw. Target: log₁₀(declared value)."
    )
    c = st.columns(4)
    c[0].metric("R² (log value)", f"{best['r2_log']:.3f}")
    c[1].metric("Median error", f"{best['median_ape']:.1f}%")
    c[2].metric("Within ±25%", f"{best['within_25pct']:.0f}%")
    c[3].metric("Within 2×", f"{best['within_2x']:.0f}%")

    st.subheader(f"All models on {test_label}")
    tbl = metrics.rename(columns={"r2_log": "R² (log)", "median_ape": "Median % error", "within_25pct": "Within ±25%",
                                  "within_2x": "Within 2×", "mae_dollars": "Mean abs error ($)",
                                  "selected": "Selected"})
    st.dataframe(tbl.style.format({"R² (log)": "{:.3f}", "Median % error": "{:.1f}%", "Within ±25%": "{:.0f}%",
                                   "Within 2×": "{:.0f}%", "Mean abs error ($)": "${:,.0f}"}),
                 width="stretch")
    st.caption("Mean absolute error in dollars is dominated by a handful of multi-million-dollar permits; "
               "the percentage metrics describe a typical permit better.")

    l, r = st.columns(2)
    with l:
        st.subheader("Why the split matters")
        rnd = report.random_split_metrics
        comp = pd.DataFrame({"Evaluation": ["Random 80/20 split", f"Out-of-time ({test_label})"],
                             "R²": [rnd["r2_log"], best["r2_log"]]})
        fig = px.bar(comp, x="Evaluation", y="R²", color="Evaluation", color_discrete_sequence=[GREY, BLUE], text_auto=".3f")
        st.plotly_chart(style(fig.update_layout(showlegend=False, xaxis_title="")), width="stretch")
        st.caption("A random split mixes years, so the model is partly tested on the same era it trained on. "
                   "Its score is optimistic for predicting new permits.")
    with r:
        st.subheader("Error by job category")
        tp = report.test_predictions.assign(ape=lambda d: (d["predicted"] - d["value"]).abs() / d["value"] * 100)
        bc = tp.groupby("JOB_CATEGORY")["ape"].agg(["median", "size"]).query("size >= 20").sort_values("median")
        fig = px.bar(bc.reset_index(), x="median", y="JOB_CATEGORY", orientation="h", color_discrete_sequence=[AMBER],
                     hover_data={"size": True})
        st.plotly_chart(style(fig.update_layout(xaxis_title="Median % error", yaxis_title="")), width="stretch")
        st.caption("New houses are predictable from their size; renovations and commercial work vary much more.")

    st.subheader("Predicted vs actual")
    tp = report.test_predictions
    sample = tp.sample(min(2500, len(tp)), random_state=1)
    fig = px.scatter(sample, x="value", y="predicted", color="JOB_CATEGORY", opacity=0.5, log_x=True, log_y=True)
    lim = [sample[["value", "predicted"]].min().min(), sample[["value", "predicted"]].max().max()]
    fig.add_trace(go.Scatter(x=lim, y=lim, mode="lines", name="Perfect", line=dict(color=RED, dash="dash")))
    st.plotly_chart(style(fig.update_layout(xaxis_title="Actual ($)", yaxis_title="Predicted ($)", legend_title=""), 460),
                    width="stretch")

    cov = report.interval_coverage
    st.subheader("Are the ranges honest?")
    st.markdown(
        f"Each prediction comes with an **{cov['level']:.0f}% range** calibrated on {sp['valid_year']} errors, separately "
        f"for each job category. On {test_label} the true value fell inside it **{cov['test_coverage']:.0f}%** of the time. "
        f"The typical range spans ÷/× **{cov['median_width_factor']:.1f}** around the estimate."
    )
    inside = (tp["value"] >= tp["lower"]) & (tp["value"] <= tp["upper"])
    cc = inside.groupby(tp["JOB_CATEGORY"]).agg(["mean", "size"]).query("size >= 20")
    fig = px.bar(cc.reset_index(), x="JOB_CATEGORY", y="mean", color_discrete_sequence=[GREEN], text_auto=".0%")
    fig.add_hline(y=cov["level"] / 100, line_dash="dash", line_color=RED, annotation_text="target")
    st.plotly_chart(style(fig.update_layout(yaxis_tickformat=".0%", yaxis_title="Coverage", xaxis_title="")),
                    width="stretch")

# ══ What drives value ════════════════════════════════════════════════════════
elif page == "What drives value":
    st.title("What drives construction value")
    imp = report.importance
    st.caption(f"Permutation importance on {test_label}: how much R² drops when one input is shuffled. "
               "Measured on unseen years, so it reflects what the model actually relies on there.")
    fig = px.bar(imp.sort_values("r2_drop"), x="r2_drop", y="label", orientation="h", error_x="r2_drop_std",
                 color_discrete_sequence=[BLUE])
    st.plotly_chart(style(fig.update_layout(xaxis_title="Drop in R² when shuffled", yaxis_title=""), 440),
                    width="stretch")

    top = imp.iloc[0]
    location = imp[imp["feature"].isin(["NEIGHBOURHOOD", "ZONING", "latitude", "longitude"])]["r2_drop"].sum()
    text_imp = imp.loc[imp["feature"] == "JOB_DESCRIPTION", "r2_drop"].sum()
    st.markdown(
        f"- **{top['label']}** matters most: shuffling it costs {top['r2_drop']:.2f} R².\n"
        f"- The **free-text job description** is worth {text_imp:.2f} R² on top of the structured fields.\n"
        f"- **Location** (neighbourhood, zoning, coordinates) costs {location:.3f} R² combined when shuffled. "
        "That doesn't mean location doesn't affect cost. It means once the size and type of work are known, "
        "location adds little predictive information."
    )

    st.subheader("Floor area vs value")
    has_fa = df.dropna(subset=["floor_area"])
    s = has_fa.sample(min(4000, len(has_fa)), random_state=0)
    fig = px.scatter(s, x="floor_area", y="value", color="JOB_CATEGORY", log_x=True, log_y=True, opacity=0.45)
    st.plotly_chart(style(fig.update_layout(xaxis_title="Floor area (log)", yaxis_title="Declared value ($, log)",
                                            legend_title=""), 460), width="stretch")
    st.caption(f"{df['floor_area'].isna().mean():.0%} of permits have no floor area; "
               "the model learns from missingness too, since small jobs often omit it.")

# ══ Predict a permit ═════════════════════════════════════════════════════════
elif page == "Predict a permit":
    st.title("Estimate a permit's declared value")
    cats, counts = report.meta["categories"], report.meta["category_counts"]

    st.caption("Work type, building type and zoning options are ordered by how often they occur with the chosen "
               "job category, so the defaults form a realistic permit.")

    def pick(label, col, within=None):
        # Most common levels first; when `within` is given, only levels seen together with it.
        pool = df if within is None else df[df["JOB_CATEGORY"] == within]
        options = pool[col].value_counts().index.tolist() or sorted(cats[col])
        return st.selectbox(label, options)

    l, r = st.columns(2)
    with l:
        job = pick("Job category", "JOB_CATEGORY")
        work = pick("Work type", "WORK_TYPE", job)
        building = pick("Building type", "BUILDING_TYPE", job)
        floor_area = st.number_input("Floor area in sq ft (0 = unknown)", 0.0, 500_000.0, 0.0, 50.0)
        units = st.number_input("Dwelling units added", -50, 500, 0)
    with r:
        neighbourhood = pick("Neighbourhood", "NEIGHBOURHOOD")
        zoning = pick("Zoning", "ZONING", job)
        year = st.slider("Permit year", report.meta["year_min"], report.meta["year_max"], report.meta["year_max"],
                         help="Limited to years in the data; the model cannot forecast later years.")
        month = st.slider("Month", 1, 12, 6)
    desc = st.text_area("Job description (optional, but improves accuracy)",
                        placeholder="To construct a Single Detached House with front attached Garage and Unenclosed Front Porch.")

    if st.button("Estimate value", type="primary"):
        centre = df.loc[df["NEIGHBOURHOOD"] == neighbourhood, ["latitude", "longitude"]].median()
        row = pd.DataFrame([{
            "JOB_CATEGORY": job, "WORK_TYPE": work, "BUILDING_TYPE": building, "ZONING": zoning,
            "NEIGHBOURHOOD": neighbourhood, "floor_area": floor_area or np.nan,
            "desc_area": description_area(desc), "YEAR": year, "MONTH_NUMBER": month,
            "units_added": units, "latitude": centre["latitude"], "longitude": centre["longitude"],
            "JOB_DESCRIPTION": desc or "",
        }])[ALL_FEATURES]
        out = predict_with_interval(report, row).iloc[0]
        c = st.columns(3)
        c[0].metric("Estimate", money(out["predicted"]))
        c[1].metric(f"{report.intervals.level:.0%} range — low", money(out["lower"]))
        c[2].metric(f"{report.intervals.level:.0%} range — high", money(out["upper"]))
        st.caption("The estimate is a typical (median-like) value for permits like this. Ranges are calibrated on "
                   "held-out years per job category; the range is wider where past predictions were less accurate.")
        n_job = counts["JOB_CATEGORY"].get(job, 0)
        if n_job < 200:
            st.warning(f"Only {n_job} permits in '{job}' — treat this estimate with extra caution.")
        if not desc:
            st.info("Adding the job description usually tightens the estimate.")

# ══ Limitations ══════════════════════════════════════════════════════════════
else:
    st.title("Limitations")
    rnd = report.random_split_metrics
    tp = report.test_predictions
    by_job = ((tp["predicted"] - tp["value"]).abs() / tp["value"] * 100).groupby(tp["JOB_CATEGORY"]).agg(["median", "size"])
    by_job = by_job[by_job["size"] >= 20].sort_values("median")
    easiest, hardest = by_job.index[0], by_job.index[-1]
    st.markdown(f"""
- **Declared, not actual, cost.** The target is the value applicants write on the permit. It is self-reported,
  often rounded, and may be understated. The model predicts what will be declared, not what the project will cost.
- **Range of validity.** Permits under {money(MIN_VALUE)} or over {money(MAX_VALUE)} were removed as placeholders or
  outliers. Estimates outside that range are not supported.
- **The world shifts.** Job categories were reorganised after 2023, and some fields (e.g. units added) are missing
  more often in recent years. Out-of-time R² is {best['r2_log']:.2f} versus {rnd['r2_log']:.2f} on a random split.
  Expect accuracy to keep drifting unless the model is retrained on recent permits.
- **No forecasting.** Tree models can't extrapolate a time trend, and values are in nominal dollars, not adjusted
  for inflation. Predictions are for the conditions of the most recent years in the data.
- **Uneven accuracy.** Median error on {test_label} ranges from {by_job.loc[easiest, 'median']:.0f}% for
  '{easiest}' to {by_job.loc[hardest, 'median']:.0f}% for '{hardest}'. Ranges are correspondingly wider for the
  harder categories.
- **Range coverage is approximate.** The {report.intervals.level:.0%} ranges came out at
  {report.interval_coverage['test_coverage']:.0f}% coverage on {test_label}; the shift between years makes exact
  coverage impossible to guarantee.
- **Importance is not causation.** Permutation importance says what the model relies on, not what makes
  construction expensive. Correlated inputs (floor area and description dimensions; neighbourhood and coordinates)
  share credit.
- **Sample vs full data.** The repository ships a {len(df):,}-permit sample when using the default file.
  Results on the full export will differ somewhat.
""")
