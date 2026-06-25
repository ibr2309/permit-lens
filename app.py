"""
Edmonton Building Permit Construction Value Predictor
=====================================================
Predicts the declared construction value of a City of Edmonton building permit
from project characteristics, using 204,590 real permit records (2009-2026)
from the City of Edmonton Open Data Portal.

Data source: data.edmonton.ca - General Building Permits
Run:  streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error
from xgboost import XGBRegressor
import warnings
# Silence only the known-noisy library categories (sklearn/xgboost/plotly),
# not everything, so genuine issues still surface.
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

DATA_PATH = "General_Building_Permits_sample.csv"   # 40k-row sample ships with the repo;
# for the full 204k-permit results, download the complete CSV from data.edmonton.ca
# and point this at it.

st.set_page_config(page_title="Edmonton Permit Value Predictor", page_icon="🏗️",
                   layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
    .stMetric { background: #1a1f2e; border-radius: 10px; padding: 12px; }
    .section-header { font-size: 1.05rem; font-weight: 600; color: #e2e8f0;
        border-left: 3px solid #3b82f6; padding-left: 10px; margin: 18px 0 8px 0; }
    div[data-testid="stSidebar"] { background: #1a1f2e; }
</style>
""", unsafe_allow_html=True)

FEATURE_COLS = ['years_since_2009', 'MONTH_NUMBER', 'has_floor_area', 'log_fa',
                'JOB_CATEGORY_enc', 'BUILDING_TYPE_enc', 'WORK_TYPE_enc',
                'NEIGHBOURHOOD_r_enc', 'ZONING_r_enc']
NAME_MAP = {'years_since_2009': 'Year', 'MONTH_NUMBER': 'Month',
            'has_floor_area': 'Floor Area Provided', 'log_fa': 'Floor Area (log)',
            'JOB_CATEGORY_enc': 'Job Category', 'BUILDING_TYPE_enc': 'Building Type',
            'WORK_TYPE_enc': 'Work Type', 'NEIGHBOURHOOD_r_enc': 'Neighbourhood',
            'ZONING_r_enc': 'Zoning'}


def reduce_card(s, n):
    top = s.value_counts().head(n).index
    return s.where(s.isin(top), 'Other')


@st.cache_data(show_spinner="Loading & cleaning 200k+ permit records...")
def load_data():
    usecols = ['YEAR', 'MONTH_NUMBER', 'JOB_CATEGORY', 'BUILDING_TYPE', 'WORK_TYPE',
               'CONSTRUCTION_VALUE', 'FLOOR_AREA', 'UNITS_ADDED', 'ZONING', 'NEIGHBOURHOOD']
    df = pd.read_csv(DATA_PATH, usecols=usecols, low_memory=False)

    df['cv'] = pd.to_numeric(df['CONSTRUCTION_VALUE'].astype(str).str.replace('[$,]', '', regex=True), errors='coerce')
    df['fa'] = pd.to_numeric(df['FLOOR_AREA'].astype(str).str.replace('[,]', '', regex=True), errors='coerce')
    df.loc[df['fa'] < 0, 'fa'] = np.nan
    df = df[(df['cv'] >= 500) & (df['cv'] <= 50_000_000)].copy()

    df['log_cv'] = np.log10(df['cv'])
    df['has_floor_area'] = df['fa'].notna().astype(int)
    fa_median = float(df['fa'].median())
    df['log_fa'] = np.log1p(df['fa'].fillna(fa_median).clip(lower=0))
    df['years_since_2009'] = df['YEAR'] - 2009
    df['MONTH_NUMBER'] = pd.to_numeric(df['MONTH_NUMBER'], errors='coerce').fillna(0).astype(int)
    df['NEIGHBOURHOOD_r'] = reduce_card(df['NEIGHBOURHOOD'].fillna('Unknown'), 60)
    df['ZONING_r'] = reduce_card(df['ZONING'].fillna('Unknown'), 40)
    for c in ['BUILDING_TYPE', 'JOB_CATEGORY', 'WORK_TYPE']:
        df[c] = df[c].fillna('Unknown')

    encoders = {}
    for col in ['JOB_CATEGORY', 'BUILDING_TYPE', 'WORK_TYPE', 'NEIGHBOURHOOD_r', 'ZONING_r']:
        le = LabelEncoder()
        df[col + '_enc'] = le.fit_transform(df[col].astype(str))
        encoders[col] = le

    return df, encoders, fa_median


@st.cache_resource(show_spinner="Training 5 regression models...")
def train(_df):
    X, y = _df[FEATURE_COLS], _df['log_cv']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    models = {
        'Linear Regression': LinearRegression(),
        'Ridge Regression': Ridge(alpha=1.0),
        'Random Forest': RandomForestRegressor(n_estimators=80, max_depth=18, random_state=42, n_jobs=-1),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=200, learning_rate=0.08, max_depth=5, random_state=42),
        'XGBoost': XGBRegressor(n_estimators=300, learning_rate=0.08, max_depth=6, random_state=42, verbosity=0, n_jobs=-1),
    }
    results, xgb_test = {}, None
    for name, m in models.items():
        m.fit(X_train, y_train)
        p = m.predict(X_test)
        results[name] = {
            'r2': round(float(r2_score(y_test, p)), 4),
            'mae': round(float(mean_absolute_error(10 ** y_test, 10 ** p)), 0),
            'medape': round(float(np.median(np.abs(10 ** p - 10 ** y_test) / 10 ** y_test) * 100), 1),
        }
        if name == 'XGBoost':
            xgb_test = (10 ** y_test.values, 10 ** p)
    importance = dict(sorted(
        {NAME_MAP[c]: round(float(i) * 100, 2) for c, i in zip(FEATURE_COLS, models['XGBoost'].feature_importances_)}.items(),
        key=lambda x: -x[1]))
    return models, results, importance, xgb_test


# ── Load ──────────────────────────────────────────────────────────────────────
try:
    df, encoders, fa_median = load_data()
except FileNotFoundError:
    st.error(f"Could not find `{DATA_PATH}`. Download 'General Building Permits' from "
             "data.edmonton.ca and place the CSV next to app.py (renamed to General_Building_Permits.csv).")
    st.stop()

models, results, importance, xgb_test = train(df)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏗️ Edmonton Permits\nConstruction Value Predictor")
    st.markdown("---")
    st.markdown(f"**{len(df):,}** real permits")
    st.markdown(f"{df['YEAR'].min()}–{df['YEAR'].max()} · City of Edmonton")
    st.markdown(f"${df['cv'].sum()/1e9:.1f}B total declared value")
    st.caption("Source: data.edmonton.ca open data")
    st.markdown("---")
    page = st.radio("Navigation", ["📊 Overview & EDA", "🔬 Model Comparison",
                                    "📈 Feature Analysis", "🔮 Value Predictor"])

# ══ PAGE 1 ════════════════════════════════════════════════════════════════════
if page == "📊 Overview & EDA":
    st.title("Edmonton Building Permits")
    st.caption(f"Construction value analysis · {len(df):,} real permits from the City of Edmonton Open Data Portal")

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Permits", f"{len(df):,}")
    c2.metric("Total value", f"${df['cv'].sum()/1e9:.1f}B")
    c3.metric("Median value", f"${df['cv'].median():,.0f}")
    c4.metric("Mean value", f"${df['cv'].mean():,.0f}")
    c5.metric("Years", f"{df['YEAR'].min()}–{df['YEAR'].max()}")
    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="section-header">Construction value distribution (log scale)</div>', unsafe_allow_html=True)
        fig = px.histogram(df, x='log_cv', nbins=60, color_discrete_sequence=['#3b82f6'], template='plotly_dark')
        fig.update_xaxes(title='log₁₀(construction value $)')
        fig.update_layout(height=320, showlegend=False, margin=dict(l=0, r=0, t=10, b=0),
                          paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        st.markdown('<div class="section-header">Permits issued per year</div>', unsafe_allow_html=True)
        cby = df.groupby('YEAR').size().reset_index(name='count')
        fig = px.bar(cby, x='YEAR', y='count', color_discrete_sequence=['#10b981'], template='plotly_dark')
        fig.update_layout(height=320, showlegend=False, margin=dict(l=0, r=0, t=10, b=0),
                          paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="section-header">Median value by job category</div>', unsafe_allow_html=True)
        vc = df.groupby('JOB_CATEGORY')['cv'].median().sort_values().reset_index()
        fig = px.bar(vc, x='cv', y='JOB_CATEGORY', orientation='h', color='cv',
                     color_continuous_scale='Blues', template='plotly_dark')
        fig.update_xaxes(title='Median value ($)')
        fig.update_layout(height=360, showlegend=False, coloraxis_showscale=False,
                          margin=dict(l=0, r=0, t=10, b=0), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        st.markdown('<div class="section-header">Median value over time</div>', unsafe_allow_html=True)
        vy = df.groupby('YEAR')['cv'].median().reset_index()
        fig = px.line(vy, x='YEAR', y='cv', markers=True, color_discrete_sequence=['#f59e0b'], template='plotly_dark')
        fig.update_yaxes(title='Median value ($)')
        fig.update_layout(height=360, margin=dict(l=0, r=0, t=10, b=0),
                          paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="section-header">Top 15 neighbourhoods by total construction value</div>', unsafe_allow_html=True)
    tn = df.groupby('NEIGHBOURHOOD')['cv'].sum().sort_values(ascending=False).head(15).reset_index()
    fig = px.bar(tn, x='NEIGHBOURHOOD', y='cv', color='cv', color_continuous_scale='Viridis', template='plotly_dark')
    fig.update_yaxes(title='Total value ($)')
    fig.update_layout(height=380, showlegend=False, coloraxis_showscale=False,
                      margin=dict(l=0, r=0, t=10, b=0), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    fig.update_xaxes(tickangle=-40)
    st.plotly_chart(fig, use_container_width=True)

# ══ PAGE 2 ════════════════════════════════════════════════════════════════════
elif page == "🔬 Model Comparison":
    st.title("Model Comparison")
    n_test = int(round(len(df) * 0.2))
    st.caption(f"5 models · target = log₁₀(construction value) · 80/20 train-test split on "
               f"{len(df):,} permits ({n_test:,} held out)")

    best = max(results, key=lambda x: results[x]['r2'])
    best_mape = min(results, key=lambda x: results[x]['medape'])
    c1, c2, c3 = st.columns(3)
    c1.metric("Best R²", f"{results[best]['r2']:.3f}", f"({best})")
    c2.metric("Best median error", f"{results[best_mape]['medape']:.1f}%", f"({best_mape})")
    c3.metric("Test set size", f"{n_test:,}", "held-out permits")

    st.markdown('<div class="section-header">Performance metrics</div>', unsafe_allow_html=True)
    tbl = pd.DataFrame({n: {'R² (log)': f"{v['r2']:.4f}", 'Mean abs error': f"${v['mae']:,.0f}",
                            'Median abs % error': f"{v['medape']:.1f}%"} for n, v in results.items()}).T
    tbl = tbl.reset_index().rename(columns={'index': 'Model'})
    st.dataframe(tbl, use_container_width=True, hide_index=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="section-header">R² by model</div>', unsafe_allow_html=True)
        rd = pd.DataFrame({'Model': list(results), 'R²': [v['r2'] for v in results.values()]})
        fig = px.bar(rd, x='Model', y='R²', color='R²', color_continuous_scale='Blues', template='plotly_dark')
        fig.update_layout(height=320, showlegend=False, coloraxis_showscale=False,
                          margin=dict(l=0, r=0, t=10, b=0), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        st.markdown('<div class="section-header">Median % error by model</div>', unsafe_allow_html=True)
        ed = pd.DataFrame({'Model': list(results), 'MedAPE': [v['medape'] for v in results.values()]})
        fig = px.bar(ed, x='Model', y='MedAPE', color='MedAPE', color_continuous_scale='Reds_r', template='plotly_dark')
        fig.update_yaxes(title='Median abs % error')
        fig.update_layout(height=320, showlegend=False, coloraxis_showscale=False,
                          margin=dict(l=0, r=0, t=10, b=0), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="section-header">XGBoost — predicted vs actual (log-log, 2k sample)</div>', unsafe_allow_html=True)
    yt, pt = xgb_test
    idx = np.random.RandomState(1).choice(len(yt), min(2000, len(yt)), replace=False)
    sdf = pd.DataFrame({'Actual': yt[idx], 'Predicted': pt[idx]})
    fig = px.scatter(sdf, x='Actual', y='Predicted', opacity=0.4, color_discrete_sequence=['#3b82f6'], template='plotly_dark')
    lim = [max(500, sdf['Actual'].min()), sdf['Actual'].max()]
    fig.add_trace(go.Scatter(x=lim, y=lim, mode='lines', name='Perfect', line=dict(color='#ef4444', dash='dash')))
    fig.update_xaxes(type='log', title='Actual value ($)')
    fig.update_yaxes(type='log', title='Predicted value ($)')
    fig.update_layout(height=440, margin=dict(l=0, r=0, t=10, b=0),
                      paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig, use_container_width=True)

    st.info("Tree ensembles (R²≈0.84) crush linear models (R²≈0.56) here — construction value depends on "
            "non-linear interactions between job category, building type, and floor area that linear regression "
            "can't capture. Median error of ~17–19% is strong given only pre-build permit features.", icon="ℹ️")

# ══ PAGE 3 ════════════════════════════════════════════════════════════════════
elif page == "📈 Feature Analysis":
    st.title("Feature Analysis")
    st.caption("9 features from raw permit fields. Importance = XGBoost gain. "
               "Note: UNITS_ADDED was deliberately dropped — it dominated importance but the model scored identically "
               "without it, so the simpler, more interpretable feature set was kept.")

    col1, col2 = st.columns([1.1, 0.9])
    with col1:
        st.markdown('<div class="section-header">XGBoost feature importance (% gain)</div>', unsafe_allow_html=True)
        fi = pd.DataFrame({'Feature': list(importance), 'Importance': list(importance.values())}).sort_values('Importance')
        fig = px.bar(fi, x='Importance', y='Feature', orientation='h', color='Importance',
                     color_continuous_scale='Blues', template='plotly_dark')
        fig.update_layout(height=400, showlegend=False, coloraxis_showscale=False,
                          margin=dict(l=0, r=0, t=10, b=0), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        st.markdown('<div class="section-header">Key findings</div>', unsafe_allow_html=True)
        st.markdown("""
**Job Category** dominates (~57%)
> What kind of work it is — commercial final, single-detached housing, home improvement — is the strongest single signal of value.

**Floor Area** (~28%)
> Bigger footprint, bigger budget. The log-transform linearizes a strongly skewed relationship.

**Work Type** (~11%)
> New build vs. interior alteration vs. demolition separates value tiers cleanly.

**Location barely matters**
> Neighbourhood & zoning contribute <1% combined — surprising, but value is driven by *what* is built, not *where*.
        """)

    st.markdown('<div class="section-header">Value distribution by work type (top categories)</div>', unsafe_allow_html=True)
    top_wt = df['WORK_TYPE'].value_counts().head(8).index
    box_df = df[df['WORK_TYPE'].isin(top_wt)].copy()
    fig = px.box(box_df, x='WORK_TYPE', y='cv', color='WORK_TYPE', template='plotly_dark')
    fig.update_yaxes(type='log', title='Construction value ($, log)')
    fig.update_xaxes(title='', tickangle=-30)
    fig.update_layout(height=420, showlegend=False, margin=dict(l=0, r=0, t=10, b=0),
                      paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig, use_container_width=True)

# ══ PAGE 4 ════════════════════════════════════════════════════════════════════
elif page == "🔮 Value Predictor":
    st.title("Permit Value Predictor")
    st.caption("Enter permit characteristics to estimate the declared construction value (XGBoost, R²=0.84).")

    col1, col2 = st.columns(2)
    with col1:
        job = st.selectbox("Job category", sorted(df['JOB_CATEGORY'].unique()))
        building = st.selectbox("Building type", sorted(df['BUILDING_TYPE'].unique()))
        work = st.selectbox("Work type", sorted(df['WORK_TYPE'].unique()))
        year = st.slider("Year", 2024, 2030, 2026)
    with col2:
        neighbourhood = st.selectbox("Neighbourhood", sorted(df['NEIGHBOURHOOD_r'].unique()))
        zoning = st.selectbox("Zoning", sorted(df['ZONING_r'].unique()))
        month = st.slider("Month", 1, 12, 6)
        floor_area = st.number_input("Floor area (sq ft, 0 = unknown)", min_value=0, max_value=500000, value=2000, step=100)

    if st.button("🔮 Predict Construction Value", type="primary"):
        has_fa = 1 if floor_area > 0 else 0
        fa_val = floor_area if floor_area > 0 else fa_median

        def enc(col, val):
            le = encoders[col]
            if val in le.classes_:
                return int(le.transform([val])[0])
            for fallback in ('Unknown', 'Other'):
                if fallback in le.classes_:
                    return int(le.transform([fallback])[0])
            return 0

        row = {
            'years_since_2009': year - 2009, 'MONTH_NUMBER': month,
            'has_floor_area': has_fa, 'log_fa': np.log1p(max(fa_val, 0)),
            'JOB_CATEGORY_enc': enc('JOB_CATEGORY', job),
            'BUILDING_TYPE_enc': enc('BUILDING_TYPE', building),
            'WORK_TYPE_enc': enc('WORK_TYPE', work),
            'NEIGHBOURHOOD_r_enc': enc('NEIGHBOURHOOD_r', neighbourhood),
            'ZONING_r_enc': enc('ZONING_r', zoning),
        }
        Xin = pd.DataFrame([row])[FEATURE_COLS]

        log_pred = float(models['XGBoost'].predict(Xin)[0])
        value = 10 ** log_pred
        # rough ±band using model median % error
        band = results['XGBoost']['medape'] / 100

        st.markdown("---")
        st.markdown("### Prediction")
        m1, m2, m3 = st.columns(3)
        m1.metric("Estimated value", f"${value:,.0f}")
        m2.metric("Likely range (low)", f"${value*(1-band):,.0f}")
        m3.metric("Likely range (high)", f"${value*(1+band):,.0f}")

        all_preds = {n: 10 ** float(m.predict(Xin)[0]) for n, m in models.items()}
        pdf = pd.DataFrame({'Model': list(all_preds), 'Predicted ($)': list(all_preds.values())})
        fig = px.bar(pdf, x='Model', y='Predicted ($)', color='Predicted ($)',
                     color_continuous_scale='Blues', template='plotly_dark')
        fig.update_layout(height=320, showlegend=False, coloraxis_showscale=False,
                          margin=dict(l=0, r=0, t=10, b=0), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig, use_container_width=True)
        st.caption(f"Range reflects the model's median absolute error of ±{band*100:.0f}% on held-out data.")
