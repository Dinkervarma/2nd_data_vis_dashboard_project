"""
AI Adoption & Corporate Performance Dashboard  —  Unified Edition
===================================================================
A single Streamlit app that merges all previous dashboard phases:

    Phase 2 (ai_adoption_dashboard.py) -> KPI cards, industry/country bars,
                                           adoption heatmap
    Phase 3 (app.py + data_utils.py)   -> region/company-size derivation,
                                           correlation matrix, rule-based
                                           insight engine
    Phase 4 (app_1.py)                 -> multi-page navigation, engineered
                                           ROI metrics, linear-regression
                                           "Predictive Insights" page
    Phase 5 (app_4.py)                 -> polished Industry Deep Dive page
                                           (KPI deltas, radar chart, heatmap)

One dataset, one filter panel, one file to deploy.

Run with:
    streamlit run app.py

Expects a CSV named "corporate_dataset.csv" (or
"corporate_ai_adoption_dataset.csv") in the same folder. If neither is
found, a file-uploader appears so the dataset can be supplied manually.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

# ═══════════════════════════════════════════════════════════════════════
# PAGE CONFIG  (must be the first Streamlit call)
# ═══════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="AI Adoption & Corporate Performance Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ═══════════════════════════════════════════════════════════════════════
# THEME / COLOR PALETTE
# ═══════════════════════════════════════════════════════════════════════
ACCENT = "#6C5CE7"
ACCENT_DARK = "#4834D4"
ACCENT_LIGHT = "#A29BFE"
NEUTRAL = "#DCDCEA"
GOOD = "#00B894"
BAD = "#D63031"
BG_CARD = "#FFFFFF"
BG_SOFT = "#F6F5FC"
TEXT_MAIN = "#2D2B4E"
TEXT_MUTED = "#7A7898"
PLOTLY_TEMPLATE = "plotly_white"
COLOR_SEQUENCE = px.colors.qualitative.Bold

st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    html, body, [class*="css"] {{ font-family: 'Inter', sans-serif; }}
    .main {{ background: linear-gradient(180deg, {BG_SOFT} 0%, #FFFFFF 320px); }}

    .hero {{
        background: linear-gradient(135deg, {ACCENT_DARK} 0%, {ACCENT} 55%, {ACCENT_LIGHT} 100%);
        border-radius: 20px; padding: 2rem 2.2rem; margin-bottom: 1.6rem;
        box-shadow: 0 10px 30px rgba(108, 92, 231, 0.25); color: white;
    }}
    .hero h1 {{ margin: 0; font-size: 2rem; font-weight: 800; letter-spacing: -0.02em; color: white; }}
    .hero p {{ margin: 0.35rem 0 0 0; font-size: 1rem; color: rgba(255,255,255,0.88); }}
    .hero .badge {{
        display: inline-block; background: rgba(255,255,255,0.18);
        border: 1px solid rgba(255,255,255,0.35); padding: 0.25rem 0.8rem;
        border-radius: 999px; font-size: 0.8rem; font-weight: 600;
        margin-top: 0.9rem; margin-right: 0.4rem; color: white;
    }}

    .section-title {{
        font-size: 1.25rem; font-weight: 700; color: {TEXT_MAIN};
        margin: 1.6rem 0 0.6rem 0; display: flex; align-items: center; gap: 0.5rem;
    }}
    .section-sub {{ color: {TEXT_MUTED}; font-size: 0.92rem; margin-bottom: 0.9rem; }}

    div[data-testid="stMetric"], .kpi-card {{
        background: {BG_CARD}; border-radius: 16px; padding: 1.1rem 1.2rem;
        border: 1px solid #ECEBF7; box-shadow: 0 4px 14px rgba(108, 92, 231, 0.06);
        height: 100%; transition: transform 0.15s ease;
    }}
    div[data-testid="stMetric"]:hover, .kpi-card:hover {{
        transform: translateY(-2px); box-shadow: 0 8px 22px rgba(108, 92, 231, 0.14);
    }}
    div[data-testid="stMetric"] label {{
        color: {TEXT_MUTED} !important; font-weight: 600; font-size: 0.8rem;
        text-transform: uppercase; letter-spacing: 0.03em;
    }}
    div[data-testid="stMetricValue"] {{ color: {TEXT_MAIN}; font-weight: 800; }}

    .kpi-label {{ font-size: 0.8rem; font-weight: 600; color: {TEXT_MUTED};
        text-transform: uppercase; letter-spacing: 0.03em; margin-bottom: 0.35rem; }}
    .kpi-value {{ font-size: 1.55rem; font-weight: 800; color: {TEXT_MAIN}; line-height: 1.15; }}
    .kpi-delta-pos {{ color: {GOOD}; font-weight: 700; font-size: 0.85rem; margin-top: 0.3rem; }}
    .kpi-delta-neg {{ color: {BAD}; font-weight: 700; font-size: 0.85rem; margin-top: 0.3rem; }}

    .insight-card {{
        background: {BG_CARD}; border-left: 4px solid {ACCENT}; border-radius: 10px;
        padding: 0.85rem 1.1rem; margin-bottom: 0.6rem;
        box-shadow: 0 2px 8px rgba(108, 92, 231, 0.06); font-size: 0.95rem; color: {TEXT_MAIN};
    }}
    .chart-card {{
        background: {BG_CARD}; border-radius: 16px; padding: 1rem 1.1rem 0.4rem 1.1rem;
        border: 1px solid #ECEBF7; box-shadow: 0 4px 14px rgba(108, 92, 231, 0.06); margin-bottom: 1.2rem;
    }}

    section[data-testid="stSidebar"] {{ background: linear-gradient(180deg, #2D2B4E 0%, #4834D4 100%); }}
    section[data-testid="stSidebar"] * {{ color: #EDEBFA !important; }}

    div[data-testid="stButton"] > button {{
        border-radius: 10px; border: 1px solid #ECEBF7; font-weight: 600;
        color: {TEXT_MAIN}; background: white;
    }}
    div[data-testid="stButton"] > button:hover {{ border-color: {ACCENT}; color: {ACCENT_DARK}; }}

    hr {{ border-color: #ECEBF7 !important; }}
    footer {{ visibility: hidden; }}
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════════
REQUIRED_COLUMNS = [
    "company_id", "industry", "country", "year", "ai_adoption_level",
    "ai_investment_usd", "automation_rate", "cost_savings", "revenue_impact",
    "productivity_gain", "employee_ai_training_hours", "ai_maturity_score",
    "deployment_count",
]
NUMERIC_COLUMNS = [
    "year", "ai_adoption_level", "ai_investment_usd", "automation_rate",
    "cost_savings", "revenue_impact", "productivity_gain",
    "employee_ai_training_hours", "ai_maturity_score", "deployment_count",
]
CANDIDATE_FILENAMES = [
    "corporate_dataset.csv",
    "corporate_ai_adoption_dataset.csv",
]
CORR_COLUMNS = [
    "ai_adoption_level", "ai_investment_usd", "automation_rate", "cost_savings",
    "revenue_impact", "productivity_gain", "employee_ai_training_hours",
    "ai_maturity_score", "deployment_count", "roi_score",
    "investment_efficiency", "productivity_index",
]
# Country -> Region mapping (derived; not present in the source data)
COUNTRY_TO_REGION = {
    "United States": "North America", "Canada": "North America",
    "Brazil": "South America",
    "United Kingdom": "Europe", "Germany": "Europe", "France": "Europe",
    "Netherlands": "Europe", "Sweden": "Europe",
    "China": "Asia Pacific", "Japan": "Asia Pacific", "India": "Asia Pacific",
    "Singapore": "Asia Pacific", "South Korea": "Asia Pacific", "Australia": "Asia Pacific",
    "UAE": "Middle East",
}
INDUSTRY_METRICS = {
    "ai_investment_usd": "Total AI Investment",
    "revenue_growth_pct": "Revenue Growth (%)",
    "cost_savings_pct": "Cost Savings (%)",
    "productivity_gain_pct": "Productivity Increase (%)",
    "ai_maturity_score": "AI Maturity Score",
    "automation_rate_pct": "Automation Rate (%)",
}
KPI_ICONS = {
    "ai_investment_usd": "💰", "revenue_growth_pct": "📈", "cost_savings_pct": "💵",
    "productivity_gain_pct": "⚙️", "ai_maturity_score": "🧠", "automation_rate_pct": "🤖",
}

PAGES = [
    "Executive Overview",
    "Financial Impact",
    "Workforce & Productivity",
    "Industry Deep Dive",
    "Predictive Insights",
    "Data Explorer",
]

# ═══════════════════════════════════════════════════════════════════════
# DATA LOADING & PREPARATION
# ═══════════════════════════════════════════════════════════════════════
def find_local_dataset() -> str | None:
    here = Path(__file__).parent
    for name in CANDIDATE_FILENAMES:
        candidate = here / name
        if candidate.exists():
            return str(candidate)
        if os.path.exists(name):
            return name
    return None


@st.cache_data(show_spinner="Loading dataset...")
def load_raw(file_source) -> tuple[pd.DataFrame | None, list[str]]:
    """Load the CSV, validate required columns, coerce numerics."""
    df = pd.read_csv(file_source)
    df.columns = [c.strip() for c in df.columns]

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        return None, missing

    for col in NUMERIC_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["industry", "country", "year"])
    for col in NUMERIC_COLUMNS:
        if df[col].isnull().any():
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val if not np.isnan(median_val) else 0)

    df["year"] = df["year"].astype(int)
    for col in ["industry", "country", "company_id"]:
        df[col] = df[col].astype(str).str.strip()
    df = df.drop_duplicates()

    return df, []


@st.cache_data(show_spinner=False)
def enrich(df: pd.DataFrame) -> pd.DataFrame:
    """Add every derived / engineered column used anywhere in the dashboard."""
    df = df.copy()

    # --- Region (derived from country) ---
    df["region"] = df["country"].map(COUNTRY_TO_REGION).fillna("Other")

    # --- Company size (proxy, derived from AI-investment quartiles) ---
    try:
        df["company_size"] = pd.qcut(
            df["ai_investment_usd"], q=4,
            labels=["Small", "Mid-size", "Large", "Enterprise"],
        )
    except ValueError:
        df["company_size"] = "Unclassified"

    # --- Adoption bands (categorical filter, 4 buckets) ---
    df["adoption_band"] = pd.cut(
        df["ai_adoption_level"],
        bins=[-0.01, 0.25, 0.5, 0.75, 1.0],
        labels=["Low (0-25%)", "Moderate (25-50%)", "High (50-75%)", "Very High (75-100%)"],
    ).astype(str)

    # --- Engineered performance metrics ---
    df["roi_score"] = df["revenue_impact"] + df["cost_savings"]
    safe_investment = df["ai_investment_usd"].replace(0, np.nan)
    df["investment_efficiency"] = (df["revenue_impact"] / safe_investment).fillna(0)
    df["productivity_index"] = df["productivity_gain"] * df["automation_rate"]

    return df


def load_dataset() -> pd.DataFrame:
    local_path = find_local_dataset()
    if local_path is not None:
        raw_df, missing_cols = load_raw(local_path)
    else:
        st.sidebar.warning("Dataset file not found locally.")
        uploaded_file = st.sidebar.file_uploader(
            "Upload corporate_dataset.csv", type="csv"
        )
        if uploaded_file is not None:
            raw_df, missing_cols = load_raw(uploaded_file)
        else:
            raw_df, missing_cols = None, []

    if raw_df is None:
        if missing_cols:
            st.error("The uploaded dataset is missing required columns: " + ", ".join(missing_cols))
        else:
            st.title("AI Adoption & Corporate Performance Dashboard")
            st.info("Please upload the dataset CSV file using the sidebar to continue.")
        st.stop()

    return enrich(raw_df)


# ═══════════════════════════════════════════════════════════════════════
# FORMATTING HELPERS
# ═══════════════════════════════════════════════════════════════════════
def fmt_currency(value: float) -> str:
    if pd.isna(value):
        return "N/A"
    if abs(value) >= 1_000_000_000:
        return f"${value / 1_000_000_000:,.2f}B"
    if abs(value) >= 1_000_000:
        return f"${value / 1_000_000:,.2f}M"
    if abs(value) >= 1_000:
        return f"${value / 1_000:,.1f}K"
    return f"${value:,.0f}"


def fmt_pct(value: float) -> str:
    if pd.isna(value):
        return "N/A"
    return f"{value * 100:,.1f}%"


def section_title(icon: str, title: str, subtitle: str = ""):
    st.markdown(f'<div class="section-title">{icon} {title}</div>', unsafe_allow_html=True)
    if subtitle:
        st.markdown(f'<div class="section-sub">{subtitle}</div>', unsafe_allow_html=True)


def kpi_card(col, icon: str, label: str, value: str, delta_pct: float | None = None):
    delta_html = ""
    if delta_pct is not None:
        delta_class = "kpi-delta-pos" if delta_pct >= 0 else "kpi-delta-neg"
        arrow = "▲" if delta_pct >= 0 else "▼"
        delta_html = f'<div class="{delta_class}">{arrow} {abs(delta_pct):.1f}% vs avg</div>'
    with col:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">{icon} {label}</div>
            <div class="kpi-value">{value}</div>
            {delta_html}
        </div>
        """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════
# LOAD DATA
# ═══════════════════════════════════════════════════════════════════════
df = load_dataset()

# ═══════════════════════════════════════════════════════════════════════
# SIDEBAR — NAVIGATION + FILTERS
# ═══════════════════════════════════════════════════════════════════════
st.sidebar.markdown("## 📊 AI Adoption Dashboard")
st.sidebar.caption("Unified analytics suite — all phases in one place")
page = st.sidebar.radio("Navigate", PAGES)

st.sidebar.markdown("---")
st.sidebar.markdown("### 🎛️ Filters")
st.sidebar.caption("Filters apply instantly to every page.")

industry_options = sorted(df["industry"].unique().tolist())
region_options = sorted(df["region"].unique().tolist())
size_options = ["Small", "Mid-size", "Large", "Enterprise"]
adoption_options = ["Low (0-25%)", "Moderate (25-50%)", "High (50-75%)", "Very High (75-100%)"]
year_min, year_max = int(df["year"].min()), int(df["year"].max())

defaults = {
    "f_industry": industry_options,
    "f_region": region_options,
    "f_size": size_options,
    "f_year": (year_min, year_max),
    "f_adoption": adoption_options,
}
for key, val in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = val


def reset_filters():
    for key, val in defaults.items():
        st.session_state[key] = val


st.sidebar.multiselect("Industry", industry_options, key="f_industry")
st.sidebar.multiselect("Region", region_options, key="f_region")
st.sidebar.multiselect("Company Size", size_options, key="f_size",
                        help="Derived from AI-investment quartiles (proxy for company scale).")
st.sidebar.slider("Year range", min_value=year_min, max_value=year_max, key="f_year")
st.sidebar.multiselect("AI Adoption Level", adoption_options, key="f_adoption")
st.sidebar.button("🔄 Reset Filters", on_click=reset_filters, use_container_width=True)

st.sidebar.markdown("---")
st.sidebar.subheader("Dataset Summary")
c1, c2 = st.sidebar.columns(2)
c1.metric("Rows", f"{df.shape[0]:,}")
c2.metric("Companies", f"{df['company_id'].nunique():,}")
c3, c4 = st.sidebar.columns(2)
c3.metric("Industries", df["industry"].nunique())
c4.metric("Countries", df["country"].nunique())


def apply_filters(source_df: pd.DataFrame) -> pd.DataFrame:
    sel_industry = st.session_state["f_industry"] or industry_options
    sel_region = st.session_state["f_region"] or region_options
    sel_size = st.session_state["f_size"] or size_options
    sel_year = st.session_state["f_year"]
    sel_adoption = st.session_state["f_adoption"] or adoption_options

    return source_df[
        source_df["industry"].isin(sel_industry)
        & source_df["region"].isin(sel_region)
        & source_df["company_size"].astype(str).isin(sel_size)
        & source_df["year"].between(sel_year[0], sel_year[1])
        & source_df["adoption_band"].isin(sel_adoption)
    ]


fdf = apply_filters(df)

# ═══════════════════════════════════════════════════════════════════════
# HERO HEADER
# ═══════════════════════════════════════════════════════════════════════
st.markdown(f"""
<div class="hero">
    <h1>📊 AI Adoption &amp; Corporate Performance Dashboard</h1>
    <p>Enterprise AI investment, adoption, and impact analytics — {len(fdf):,} of {len(df):,} records in view</p>
    <span class="badge">📍 {page}</span>
    <span class="badge">🗂️ Unified · All Phases</span>
</div>
""", unsafe_allow_html=True)

if fdf.empty:
    st.warning("No data matches the current filter selection. Please broaden your filters in the sidebar.")
    st.stop()

# ═══════════════════════════════════════════════════════════════════════
# PAGE 1 — EXECUTIVE OVERVIEW
# ═══════════════════════════════════════════════════════════════════════
if page == "Executive Overview":
    section_title("📌", "Key Performance Indicators")
    k1, k2, k3, k4, k5, k6 = st.columns(6)
    k1.metric("Total Companies", f"{fdf['company_id'].nunique():,}")
    k2.metric("Total AI Investment", fmt_currency(fdf["ai_investment_usd"].sum()))
    k3.metric("Avg Productivity Gain", fmt_pct(fdf["productivity_gain"].mean()))
    k4.metric("Avg Revenue Impact", fmt_currency(fdf["revenue_impact"].mean()))
    k5.metric("Avg AI Maturity Score", f"{fdf['ai_maturity_score'].mean():.2f} / 10")
    k6.metric("Avg Automation Rate", fmt_pct(fdf["automation_rate"].mean()))

    st.markdown("---")
    section_title("🏭", "Adoption by Industry & Region")
    col1, col2 = st.columns(2)
    with col1:
        industry_adoption = (
            fdf.groupby("industry", as_index=False)["ai_adoption_level"]
            .mean().sort_values("ai_adoption_level", ascending=True)
        )
        fig = px.bar(
            industry_adoption, x="ai_adoption_level", y="industry", orientation="h",
            text=industry_adoption["ai_adoption_level"].map(lambda v: f"{v * 100:.1f}%"),
            color="ai_adoption_level", color_continuous_scale="Purples",
            title="Average AI Adoption Level by Industry",
            labels={"ai_adoption_level": "Avg Adoption Level", "industry": ""},
            template=PLOTLY_TEMPLATE,
        )
        fig.update_traces(textposition="outside", cliponaxis=False)
        fig.update_layout(xaxis_tickformat=".0%", coloraxis_showscale=False, height=420)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        region_counts = fdf.groupby("region")["company_id"].nunique().reset_index(name="companies")
        fig = px.pie(
            region_counts, names="region", values="companies", hole=0.5,
            color_discrete_sequence=COLOR_SEQUENCE,
            title="Company Distribution by Region", template=PLOTLY_TEMPLATE,
        )
        fig.update_traces(textinfo="percent+label", textfont_size=11)
        fig.update_layout(height=420)
        st.plotly_chart(fig, use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        yearly_growth = fdf.groupby("year", as_index=False)["ai_adoption_level"].mean().sort_values("year")
        fig = px.line(
            yearly_growth, x="year", y="ai_adoption_level", markers=True,
            title="Year-wise AI Adoption Growth",
            labels={"year": "Year", "ai_adoption_level": "Avg Adoption Level"},
            template=PLOTLY_TEMPLATE,
        )
        fig.update_traces(line=dict(color=ACCENT, width=3))
        fig.update_layout(yaxis_tickformat=".0%", height=420)
        st.plotly_chart(fig, use_container_width=True)

    with col4:
        heat_data = fdf.groupby(["industry", "region"], as_index=False)["ai_adoption_level"].mean()
        heat_pivot = heat_data.pivot(index="industry", columns="region", values="ai_adoption_level")
        fig = px.imshow(
            heat_pivot, color_continuous_scale="Purples", aspect="auto",
            labels=dict(color="Avg Adoption"),
            title="Adoption Heatmap: Industry vs Region", template=PLOTLY_TEMPLATE,
        )
        fig.update_layout(height=420)
        st.plotly_chart(fig, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════
# PAGE 2 — FINANCIAL IMPACT
# ═══════════════════════════════════════════════════════════════════════
elif page == "Financial Impact":
    section_title("💰", "Financial Impact")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total AI Investment", fmt_currency(fdf["ai_investment_usd"].sum()))
    k2.metric("Total Cost Savings", fmt_currency(fdf["cost_savings"].sum()))
    k3.metric("Avg Revenue Impact", fmt_currency(fdf["revenue_impact"].mean()))
    k4.metric("Avg ROI Score", fmt_currency(fdf["roi_score"].mean()))

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        sample_df = fdf.sample(min(5000, len(fdf)), random_state=42)
        fig = px.scatter(
            sample_df, x="ai_investment_usd", y="revenue_impact", color="industry",
            opacity=0.6, title="AI Investment vs Revenue Impact",
            labels={"ai_investment_usd": "AI Investment (USD)", "revenue_impact": "Revenue Impact (USD)"},
            template=PLOTLY_TEMPLATE, color_discrete_sequence=COLOR_SEQUENCE,
        )
        fig.update_layout(height=420)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        top_ind_rev = (
            fdf.groupby("industry", as_index=False)["revenue_impact"].mean()
            .sort_values("revenue_impact", ascending=False).head(10)
        )
        fig = px.bar(
            top_ind_rev, x="revenue_impact", y="industry", orientation="h",
            title="Top Industries by Avg Revenue Impact",
            labels={"revenue_impact": "Avg Revenue Impact (USD)", "industry": ""},
            template=PLOTLY_TEMPLATE, color="revenue_impact", color_continuous_scale="Greens",
        )
        fig.update_layout(yaxis={"categoryorder": "total ascending"}, coloraxis_showscale=False, height=420)
        st.plotly_chart(fig, use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        top_region_rev = (
            fdf.groupby("region", as_index=False)["revenue_impact"].mean()
            .sort_values("revenue_impact", ascending=False)
        )
        fig = px.bar(
            top_region_rev, x="revenue_impact", y="region", orientation="h",
            title="Avg Revenue Impact by Region",
            labels={"revenue_impact": "Avg Revenue Impact (USD)", "region": ""},
            template=PLOTLY_TEMPLATE, color="revenue_impact", color_continuous_scale="Purples",
        )
        fig.update_layout(yaxis={"categoryorder": "total ascending"}, coloraxis_showscale=False, height=420)
        st.plotly_chart(fig, use_container_width=True)

    with col4:
        roi_by_industry = (
            fdf.groupby("industry", as_index=False)["roi_score"].mean()
            .sort_values("roi_score", ascending=False)
        )
        fig = px.bar(
            roi_by_industry, x="industry", y="roi_score",
            title="ROI Analysis — Avg ROI Score by Industry",
            labels={"industry": "", "roi_score": "Avg ROI Score (Revenue Impact + Cost Savings)"},
            template=PLOTLY_TEMPLATE, color="roi_score", color_continuous_scale="Oranges",
        )
        fig.update_layout(xaxis_tickangle=-35, coloraxis_showscale=False, height=420)
        st.plotly_chart(fig, use_container_width=True)

    st.caption(
        "💡 **roi_score** = revenue_impact + cost_savings · "
        "**investment_efficiency** = revenue_impact / ai_investment_usd (engineered metrics)."
    )

# ═══════════════════════════════════════════════════════════════════════
# PAGE 3 — WORKFORCE & PRODUCTIVITY
# ═══════════════════════════════════════════════════════════════════════
elif page == "Workforce & Productivity":
    section_title("⚙️", "Workforce & Productivity")
    k1, k2, k3 = st.columns(3)
    k1.metric("Avg Training Hours", f"{fdf['employee_ai_training_hours'].mean():.1f} hrs")
    k2.metric("Avg Productivity Gain", fmt_pct(fdf["productivity_gain"].mean()))
    k3.metric("Avg Automation Rate", fmt_pct(fdf["automation_rate"].mean()))

    st.markdown("---")
    sample_df = fdf.sample(min(5000, len(fdf)), random_state=42)
    col1, col2 = st.columns(2)
    with col1:
        fig = px.scatter(
            sample_df, x="employee_ai_training_hours", y="ai_maturity_score", color="industry",
            opacity=0.6, title="Training Hours vs AI Maturity Score",
            labels={"employee_ai_training_hours": "Training Hours", "ai_maturity_score": "AI Maturity Score"},
            template=PLOTLY_TEMPLATE, color_discrete_sequence=COLOR_SEQUENCE,
        )
        fig.update_layout(height=420)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.scatter(
            sample_df, x="automation_rate", y="productivity_gain", color="industry",
            opacity=0.6, title="Automation Rate vs Productivity Gain",
            labels={"automation_rate": "Automation Rate", "productivity_gain": "Productivity Gain"},
            template=PLOTLY_TEMPLATE, color_discrete_sequence=COLOR_SEQUENCE,
        )
        fig.update_layout(height=420)
        st.plotly_chart(fig, use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        fig = px.scatter(
            sample_df, x="ai_adoption_level", y="productivity_gain", color="adoption_band",
            opacity=0.6, title="AI Adoption Level vs Productivity Gain",
            labels={"ai_adoption_level": "AI Adoption Level", "productivity_gain": "Productivity Gain"},
            template=PLOTLY_TEMPLATE, color_discrete_sequence=COLOR_SEQUENCE,
        )
        fig.update_layout(height=420)
        st.plotly_chart(fig, use_container_width=True)

    with col4:
        fig = px.box(
            fdf, x="industry", y="productivity_gain", color="industry",
            title="Productivity Gain Distribution by Industry",
            labels={"industry": "", "productivity_gain": "Productivity Gain"},
            template=PLOTLY_TEMPLATE,
        )
        fig.update_layout(xaxis_tickangle=-35, showlegend=False, height=420)
        st.plotly_chart(fig, use_container_width=True)

    prod_trend = fdf.groupby("year", as_index=False)["productivity_gain"].mean().sort_values("year")
    fig = px.line(
        prod_trend, x="year", y="productivity_gain", markers=True,
        title="Productivity Gain Trend by Year",
        labels={"year": "Year", "productivity_gain": "Avg Productivity Gain"},
        template=PLOTLY_TEMPLATE,
    )
    fig.update_traces(line=dict(color=ACCENT, width=3))
    fig.update_layout(height=380)
    st.plotly_chart(fig, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════
# PAGE 4 — INDUSTRY DEEP DIVE
# ═══════════════════════════════════════════════════════════════════════
elif page == "Industry Deep Dive":

    @st.cache_data(show_spinner=False)
    def compute_industry_summary(data: pd.DataFrame) -> pd.DataFrame:
        g = data.groupby("industry").agg(
            ai_investment_usd=("ai_investment_usd", "sum"),
            avg_investment=("ai_investment_usd", "mean"),
            revenue_impact=("revenue_impact", "mean"),
            cost_savings=("cost_savings", "mean"),
            productivity_gain=("productivity_gain", "mean"),
            ai_maturity_score=("ai_maturity_score", "mean"),
            automation_rate=("automation_rate", "mean"),
        ).reset_index()
        g["revenue_growth_pct"] = g["revenue_impact"] / g["avg_investment"] * 100
        g["cost_savings_pct"] = g["cost_savings"] / g["avg_investment"] * 100
        g["productivity_gain_pct"] = g["productivity_gain"] * 100
        g["automation_rate_pct"] = g["automation_rate"] * 100
        return g

    def _fmt_metric(key: str, val: float) -> str:
        if key == "ai_investment_usd":
            return fmt_currency(val)
        if key == "ai_maturity_score":
            return f"{val:.2f}/10"
        return f"{val:.1f}%"

    def _build_comparison_bars(summary: pd.DataFrame, selected: str) -> go.Figure:
        fig = make_subplots(
            rows=2, cols=3, subplot_titles=list(INDUSTRY_METRICS.values()),
            horizontal_spacing=0.10, vertical_spacing=0.20,
        )
        positions = [(1, 1), (1, 2), (1, 3), (2, 1), (2, 2), (2, 3)]
        for (r, c), (key, _label) in zip(positions, INDUSTRY_METRICS.items()):
            sorted_df = summary[["industry", key]].sort_values(key, ascending=True)
            colors = [ACCENT if ind == selected else NEUTRAL for ind in sorted_df["industry"]]
            fig.add_trace(
                go.Bar(x=sorted_df[key], y=sorted_df["industry"], orientation="h",
                       marker=dict(color=colors, line=dict(width=0)), showlegend=False,
                       hovertemplate="%{y}: %{x:,.2f}<extra></extra>"),
                row=r, col=c,
            )
        fig.update_layout(height=640, margin=dict(t=60, b=20, l=10, r=10),
                           plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                           font=dict(size=11, family="Inter, sans-serif", color=TEXT_MAIN))
        fig.update_annotations(font=dict(size=12, family="Inter, sans-serif", color=TEXT_MAIN))
        fig.update_xaxes(showgrid=True, gridcolor="#EFEEF9", zeroline=False)
        fig.update_yaxes(showgrid=False)
        return fig

    def _build_radar(summary: pd.DataFrame, selected: str) -> go.Figure:
        keys = list(INDUSTRY_METRICS.keys())
        labels = list(INDUSTRY_METRICS.values())
        mins, maxs = summary[keys].min(), summary[keys].max()
        norm = (summary[keys] - mins) / (maxs - mins).replace(0, 1) * 100
        sel_row = norm[summary["industry"] == selected].iloc[0]
        avg_row = norm.mean()
        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(
            r=list(sel_row.values) + [sel_row.values[0]], theta=labels + [labels[0]],
            fill="toself", name=selected, line=dict(color=ACCENT, width=2),
            fillcolor="rgba(108, 92, 231, 0.28)",
        ))
        fig.add_trace(go.Scatterpolar(
            r=list(avg_row.values) + [avg_row.values[0]], theta=labels + [labels[0]],
            fill="toself", name="All-Industry Average", line=dict(color="#B2B0C9", width=2, dash="dot"),
            fillcolor="rgba(178, 176, 201, 0.18)",
        ))
        fig.update_layout(
            polar=dict(bgcolor="rgba(0,0,0,0)",
                       radialaxis=dict(visible=True, range=[0, 100], gridcolor="#EFEEF9"),
                       angularaxis=dict(gridcolor="#EFEEF9")),
            showlegend=True, height=440, margin=dict(t=30, b=20, l=40, r=40),
            legend=dict(orientation="h", yanchor="bottom", y=-0.18),
            paper_bgcolor="rgba(0,0,0,0)", font=dict(family="Inter, sans-serif", color=TEXT_MAIN, size=11),
        )
        return fig

    def _build_heatmap(summary: pd.DataFrame) -> go.Figure:
        keys = list(INDUSTRY_METRICS.keys())
        labels = list(INDUSTRY_METRICS.values())
        ordered = summary.sort_values("ai_investment_usd", ascending=False)
        mins, maxs = ordered[keys].min(), ordered[keys].max()
        norm = (ordered[keys] - mins) / (maxs - mins).replace(0, 1)
        text = ordered[keys].apply(lambda col: [_fmt_metric(col.name, v) for v in col])
        fig = go.Figure(data=go.Heatmap(
            z=norm.values, x=labels, y=ordered["industry"], text=text.values,
            texttemplate="%{text}", textfont=dict(size=10, family="Inter, sans-serif"),
            colorscale="Purples", colorbar=dict(title="Relative<br>Performance"),
            hovertemplate="%{y} — %{x}: %{text}<extra></extra>", xgap=3, ygap=3,
        ))
        fig.update_layout(height=440, margin=dict(t=30, b=20, l=10, r=10),
                           paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                           font=dict(family="Inter, sans-serif", color=TEXT_MAIN, size=11))
        return fig

    def _generate_industry_insights(summary: pd.DataFrame, selected: str) -> list[str]:
        row = summary[summary["industry"] == selected].iloc[0]
        insights = []
        leader = summary.loc[summary["ai_investment_usd"].idxmax()]
        insights.append(f"💰 <b>{leader['industry']}</b> receives the highest total AI investment at {fmt_currency(leader['ai_investment_usd'])}.")
        leader = summary.loc[summary["productivity_gain_pct"].idxmax()]
        insights.append(f"⚙️ <b>{leader['industry']}</b> achieves the greatest productivity increase at {leader['productivity_gain_pct']:.1f}%.")
        leader = summary.loc[summary["ai_maturity_score"].idxmax()]
        insights.append(f"🧠 <b>{leader['industry']}</b> has the highest AI maturity score ({leader['ai_maturity_score']:.2f}/10).")
        leader = summary.loc[summary["automation_rate_pct"].idxmax()]
        insights.append(f"🤖 <b>{leader['industry']}</b> has the highest automation rate at {leader['automation_rate_pct']:.1f}%.")
        for key, label in INDUSTRY_METRICS.items():
            avg = summary[key].mean()
            diff_pct = (row[key] - avg) / avg * 100 if avg != 0 else 0
            direction = "above" if diff_pct >= 0 else "below"
            arrow = "🔼" if diff_pct >= 0 else "🔽"
            insights.append(f"{arrow} <b>{selected}</b>'s {label} is <b>{abs(diff_pct):.1f}% {direction}</b> the all-industry average.")
        return insights

    summary = compute_industry_summary(fdf)
    industries_available = sorted(summary["industry"].unique().tolist())
    default_idx = industries_available.index("Financial Services") if "Financial Services" in industries_available else 0

    section_title("🏭", "Industry Deep Dive", "Compare one industry's performance against every other, in detail.")
    selected_industry = st.selectbox("Select Industry", options=industries_available, index=default_idx)

    row = summary[summary["industry"] == selected_industry].iloc[0]
    overall_avg = {key: summary[key].mean() for key in INDUSTRY_METRICS}

    kpi_cols = st.columns(6)
    for col, (key, label) in zip(kpi_cols, INDUSTRY_METRICS.items()):
        delta_pct = (row[key] - overall_avg[key]) / overall_avg[key] * 100 if overall_avg[key] != 0 else 0
        kpi_card(col, KPI_ICONS[key], label, _fmt_metric(key, row[key]), delta_pct)

    st.write("")
    section_title("📊", "Selected Industry vs. All Others",
                   "Highlighted bars show where the selected industry ranks on every metric.")
    st.markdown('<div class="chart-card">', unsafe_allow_html=True)
    st.plotly_chart(_build_comparison_bars(summary, selected_industry), use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    section_title("🕸️", "Multi-Metric Profile & Industry Heatmap")
    col_radar, col_heat = st.columns(2)
    with col_radar:
        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        st.plotly_chart(_build_radar(summary, selected_industry), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with col_heat:
        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        st.plotly_chart(_build_heatmap(summary), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    section_title("💡", "Business Questions Answered")
    for line in _generate_industry_insights(summary, selected_industry):
        st.markdown(f'<div class="insight-card">{line}</div>', unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════
# PAGE 5 — PREDICTIVE INSIGHTS
# ═══════════════════════════════════════════════════════════════════════
elif page == "Predictive Insights":
    section_title("🔮", "Predictive Insights")
    st.caption(
        "Engineered metrics: roi_score = revenue_impact + cost_savings · "
        "investment_efficiency = revenue_impact / ai_investment_usd · "
        "productivity_index = productivity_gain × automation_rate"
    )

    corr_cols = [c for c in CORR_COLUMNS if c in fdf.columns]
    corr_df = fdf[corr_cols].corr().round(3) if len(fdf) >= 2 else pd.DataFrame()

    if not corr_df.empty:
        fig = px.imshow(
            corr_df, text_auto=".2f", aspect="auto", title="Correlation Matrix — Key AI & Performance Metrics",
            color_continuous_scale="RdBu_r", zmin=-1, zmax=1, template=PLOTLY_TEMPLATE,
        )
        fig.update_xaxes(tickangle=45)
        fig.update_layout(height=560)
        st.plotly_chart(
            fig, use_container_width=True,
            config={"displaylogo": False, "toImageButtonOptions": {"format": "png", "filename": "correlation_heatmap", "scale": 2}},
        )
        with st.expander("📥 Download correlation matrix as CSV"):
            st.download_button(
                "Download correlation_matrix.csv",
                data=corr_df.to_csv().encode("utf-8"),
                file_name="correlation_matrix.csv", mime="text/csv",
            )
        st.caption(
            "💡 Reading the heatmap: values near **+1** indicate a strong positive relationship, "
            "values near **-1** indicate a strong inverse relationship, and values near **0** indicate "
            "little to no linear relationship."
        )

    # --- Rule-based business insights ---
    st.markdown("---")
    section_title("📈", "Automated Business Insights", "Rule-based, not a black box — results stay reproducible and explainable.")

    def generate_insights(data: pd.DataFrame, corr: pd.DataFrame) -> list[str]:
        insights = []
        if data.empty:
            return ["No data matches the current filter selection."]

        by_industry = data.groupby("industry")["ai_adoption_level"].mean().sort_values(ascending=False)
        if not by_industry.empty:
            top_ind, top_val = by_industry.index[0], by_industry.iloc[0]
            insights.append(f"**{top_ind}** leads AI adoption among the filtered companies, averaging **{top_val:.1%}** adoption level.")

        if not corr.empty and corr.notna().values.any():
            corr_unstacked = corr.where(~np.eye(len(corr), dtype=bool)).unstack().dropna()
            if not corr_unstacked.empty:
                strongest = corr_unstacked.abs().idxmax()
                strongest_val = corr.loc[strongest[0], strongest[1]]
                direction = "positive" if strongest_val > 0 else "negative"
                insights.append(
                    f"**{strongest[0].replace('_', ' ').title()}** and **{strongest[1].replace('_', ' ').title()}** "
                    f"show the strongest {direction} relationship (r = {strongest_val:.2f})."
                )

        region_sums = data.groupby("region")[["revenue_impact", "ai_investment_usd"]].sum()
        region_sums = region_sums[region_sums["ai_investment_usd"] > 0]
        region_roi = (region_sums["revenue_impact"] / region_sums["ai_investment_usd"]).sort_values(ascending=False)
        if not region_roi.empty:
            insights.append(f"**{region_roi.index[0]}** delivers the best return on AI investment, generating **${region_roi.iloc[0]:.2f}** in revenue impact per $1 invested.")

        if "employee_ai_training_hours" in corr.index and "productivity_gain" in corr.columns:
            tp_corr = corr.loc["employee_ai_training_hours", "productivity_gain"]
            if pd.notna(tp_corr):
                strength = "strong" if abs(tp_corr) > 0.5 else "moderate" if abs(tp_corr) > 0.2 else "weak"
                insights.append(
                    f"Employee AI training hours show a **{strength}** correlation (r = {tp_corr:.2f}) with productivity gains, "
                    f"{'supporting continued training investment.' if tp_corr > 0 else 'suggesting training alone may not drive productivity.'}"
                )

        by_year = data.groupby("year")["ai_adoption_level"].mean()
        if len(by_year) > 1:
            change = by_year.iloc[-1] - by_year.iloc[0]
            trend = "increased" if change > 0 else "decreased"
            insights.append(f"Average AI adoption {trend} by **{abs(change):.1%}** from {by_year.index[0]} to {by_year.index[-1]} across the filtered data.")
        return insights

    insights = generate_insights(fdf, corr_df)
    ins_col1, ins_col2 = st.columns(2)
    for i, insight in enumerate(insights):
        target_col = ins_col1 if i % 2 == 0 else ins_col2
        with target_col:
            st.markdown(f'<div class="insight-card">{insight}</div>', unsafe_allow_html=True)

    # --- Linear regression ---
    st.markdown("---")
    section_title("🧮", "Linear Regression — Predicting AI Maturity Score")

    feature_cols = ["employee_ai_training_hours", "ai_investment_usd", "automation_rate", "productivity_gain"]
    target_col = "ai_maturity_score"
    reg_df = fdf[feature_cols + [target_col]].dropna()

    if len(reg_df) < 20:
        st.warning("Not enough data available for the selected filters to run a regression model.")
    else:
        X = reg_df[feature_cols]
        y = reg_df[target_col]
        X_scaled = StandardScaler().fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        r2 = r2_score(y_test, y_pred)

        col1, col2 = st.columns([1, 2])
        with col1:
            st.metric("R² Score (Test Set)", f"{r2:.3f}")
            st.metric("Training Samples", f"{len(X_train):,}")
            st.metric("Test Samples", f"{len(X_test):,}")

        with col2:
            importance_df = pd.DataFrame({"feature": feature_cols, "coefficient": model.coef_})
            importance_df["abs_coefficient"] = importance_df["coefficient"].abs()
            importance_df = importance_df.sort_values("abs_coefficient", ascending=False)
            fig = px.bar(
                importance_df, x="abs_coefficient", y="feature", orientation="h",
                title="Feature Importance — Key Factors Influencing AI Maturity Score",
                labels={"abs_coefficient": "Standardized Coefficient Magnitude", "feature": "Feature"},
                template=PLOTLY_TEMPLATE, color="coefficient", color_continuous_scale="RdBu_r",
            )
            fig.update_layout(yaxis={"categoryorder": "total ascending"}, height=340)
            st.plotly_chart(fig, use_container_width=True)

        top_factor = importance_df.iloc[0]
        direction = "positively" if top_factor["coefficient"] > 0 else "negatively"
        st.info(
            f"**{top_factor['feature']}** has the strongest association with AI maturity score, "
            f"affecting it {direction} among the selected features. The model explains "
            f"**{r2 * 100:.1f}%** of the variance in AI maturity score on the held-out test data "
            f"for the currently selected filters."
        )

# ═══════════════════════════════════════════════════════════════════════
# PAGE 6 — DATA EXPLORER
# ═══════════════════════════════════════════════════════════════════════
elif page == "Data Explorer":
    section_title("🔎", "Filtered Data Explorer")
    st.caption(f"Showing {len(fdf):,} of {len(df):,} total records based on your current filter selection.")

    display_cols = [
        "company_id", "industry", "country", "region", "company_size", "year",
        "ai_adoption_level", "adoption_band", "ai_investment_usd", "automation_rate",
        "cost_savings", "revenue_impact", "productivity_gain", "employee_ai_training_hours",
        "ai_maturity_score", "deployment_count", "roi_score", "investment_efficiency", "productivity_index",
    ]
    display_cols = [c for c in display_cols if c in fdf.columns]
    st.dataframe(fdf[display_cols], use_container_width=True, height=500)

    csv_bytes = fdf[display_cols].to_csv(index=False).encode("utf-8")
    st.download_button(
        "📥 Download filtered data as CSV", data=csv_bytes,
        file_name="filtered_ai_adoption_data.csv", mime="text/csv",
    )

# ═══════════════════════════════════════════════════════════════════════
# FOOTER
# ═══════════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown(
    '<p style="text-align:center; color:#94a3b8; font-size:0.8rem;">'
    "AI Adoption &amp; Corporate Performance Dashboard · Unified Edition · Built with Streamlit &amp; Plotly</p>",
    unsafe_allow_html=True,
)
