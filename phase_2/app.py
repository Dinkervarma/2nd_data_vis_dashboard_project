"""
Corporate AI Adoption — Executive Dashboard
Built with Streamlit + Plotly
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# ----------------------------------------------------------------------------
# PAGE CONFIG
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="Corporate AI Adoption Dashboard",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------------------------------------------------------
# CUSTOM CSS — modern, professional look
# ----------------------------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    #MainMenu, footer, header {visibility: hidden;}

    .block-container { padding-top: 1.5rem; padding-bottom: 2rem; max-width: 1400px; }

    /* Header banner */
    .main-header {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #312e81 100%);
        padding: 2rem 2.2rem;
        border-radius: 16px;
        margin-bottom: 1.6rem;
        box-shadow: 0 10px 30px rgba(15, 23, 42, 0.25);
    }
    .main-header h1 {
        color: #ffffff; font-size: 1.9rem; font-weight: 800; margin: 0;
        letter-spacing: -0.02em;
    }
    .main-header p {
        color: #c7d2fe; font-size: 0.95rem; margin-top: 0.35rem; font-weight: 400;
    }

    /* KPI cards */
    .kpi-card {
        background: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 14px;
        padding: 1.1rem 1.3rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
        height: 100%;
        border-left: 4px solid #4f46e5;
        transition: transform 0.15s ease;
    }
    .kpi-card:hover { transform: translateY(-2px); box-shadow: 0 6px 16px rgba(0,0,0,0.08); }
    .kpi-label {
        font-size: 0.76rem; font-weight: 600; color: #6b7280;
        text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 0.3rem;
    }
    .kpi-value { font-size: 1.55rem; font-weight: 800; color: #0f172a; line-height: 1.2; }
    .kpi-sub { font-size: 0.78rem; color: #10b981; font-weight: 600; margin-top: 0.25rem; }
    .kpi-sub.negative { color: #ef4444; }

    /* Section headers */
    .section-title {
        font-size: 1.25rem; font-weight: 700; color: #0f172a;
        margin: 1.6rem 0 0.8rem 0; padding-bottom: 0.5rem;
        border-bottom: 2px solid #eef2ff;
    }

    /* Insight box */
    .insight-box {
        background: linear-gradient(135deg, #eef2ff 0%, #f5f3ff 100%);
        border: 1px solid #e0e7ff;
        border-left: 4px solid #6366f1;
        border-radius: 12px;
        padding: 1rem 1.3rem;
        margin-bottom: 0.7rem;
        font-size: 0.92rem;
        color: #1e1b4b;
    }
    .insight-box b { color: #4338ca; }

    section[data-testid="stSidebar"] {
        background: #0f172a;
    }
    section[data-testid="stSidebar"] * { color: #e2e8f0 !important; }
    section[data-testid="stSidebar"] .stMultiSelect [data-baseweb="tag"] {
        background-color: #4f46e5 !important;
    }

    div[data-testid="stMetricValue"] { font-weight: 700; }
</style>
""", unsafe_allow_html=True)

PLOTLY_TEMPLATE = "plotly_white"
COLOR_SEQ = px.colors.qualitative.Bold
ACCENT = "#4f46e5"


# ----------------------------------------------------------------------------
# DATA LOADING & FEATURE ENGINEERING (mirrors the source notebook)
# ----------------------------------------------------------------------------
@st.cache_data
def load_data(file) -> pd.DataFrame:
    df = pd.read_csv(file)

    # --- Engineered features (exact logic from the notebook) ---
    df["roi_score"] = (
        (df["revenue_impact"] + df["cost_savings"] - df["ai_investment_usd"])
        / df["ai_investment_usd"]
    )
    df["investment_efficiency"] = df["revenue_impact"] / df["ai_investment_usd"]
    df["productivity_index"] = df["productivity_gain"] * df["automation_rate"]
    df["ai_effectiveness_score"] = (
        df["ai_maturity_score"] * df["ai_adoption_level"] * df["automation_rate"]
    )
    df["revenue_per_ai_dollar"] = df["revenue_impact"] / df["ai_investment_usd"]
    df["business_impact_score"] = df["revenue_impact"] + df["cost_savings"]

    return df


DEFAULT_PATH = "corporate_dataset.csv"

st.sidebar.markdown("## 🤖 AI Adoption Dashboard")
st.sidebar.caption("Corporate AI Adoption Analytics")
st.sidebar.markdown("---")

uploaded = st.sidebar.file_uploader("Upload dataset (CSV)", type=["csv"])

data_source = None
if uploaded is not None:
    data_source = uploaded
else:
    import os
    if os.path.exists(DEFAULT_PATH):
        data_source = DEFAULT_PATH

if data_source is None:
    st.markdown('<div class="main-header"><h1>🤖 Corporate AI Adoption Dashboard</h1>'
                '<p>Upload your corporate_dataset.csv to begin</p></div>', unsafe_allow_html=True)
    st.info("👈 Upload the `corporate_dataset.csv` file from the sidebar to load the dashboard.")
    st.stop()

try:
    df_raw = load_data(data_source)
except Exception as e:
    st.error(f"Could not read the dataset: {e}")
    st.stop()

required_cols = [
    "company_id", "industry", "country", "year", "ai_adoption_level",
    "ai_investment_usd", "automation_rate", "cost_savings", "revenue_impact",
    "productivity_gain", "employee_ai_training_hours", "ai_maturity_score",
    "deployment_count",
]
missing = [c for c in required_cols if c not in df_raw.columns]
if missing:
    st.error(f"The uploaded dataset is missing expected columns: {missing}")
    st.stop()

# ----------------------------------------------------------------------------
# SIDEBAR FILTERS
# ----------------------------------------------------------------------------
st.sidebar.markdown("### 🔍 Filters")

years = sorted(df_raw["year"].dropna().unique().tolist())
countries = sorted(df_raw["country"].dropna().unique().tolist())
industries = sorted(df_raw["industry"].dropna().unique().tolist())

sel_years = st.sidebar.multiselect("Year", years, default=years)
sel_countries = st.sidebar.multiselect("Country", countries, default=countries)
sel_industries = st.sidebar.multiselect("Industry", industries, default=industries)

st.sidebar.markdown("---")
page = st.sidebar.radio(
    "📑 Navigate",
    ["Executive Overview", "Financial Impact", "Operational Insights", "Advanced Analytics"],
)

df = df_raw[
    df_raw["year"].isin(sel_years if sel_years else years)
    & df_raw["country"].isin(sel_countries if sel_countries else countries)
    & df_raw["industry"].isin(sel_industries if sel_industries else industries)
].copy()

st.sidebar.markdown("---")
st.sidebar.markdown(f"**Records in view:** {len(df):,} / {len(df_raw):,}")

csv_bytes = df.to_csv(index=False).encode("utf-8")
st.sidebar.download_button(
    "⬇️ Download Filtered Data",
    data=csv_bytes,
    file_name=f"ai_adoption_filtered_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
    mime="text/csv",
    width='stretch',
)

if df.empty:
    st.markdown('<div class="main-header"><h1>🤖 Corporate AI Adoption Dashboard</h1></div>',
                unsafe_allow_html=True)
    st.warning("No data matches the selected filters. Please broaden your selection.")
    st.stop()


# ----------------------------------------------------------------------------
# HELPERS
# ----------------------------------------------------------------------------
def fmt_usd(x):
    if pd.isna(x):
        return "N/A"
    if abs(x) >= 1e9:
        return f"${x/1e9:,.2f}B"
    if abs(x) >= 1e6:
        return f"${x/1e6:,.1f}M"
    if abs(x) >= 1e3:
        return f"${x/1e3:,.1f}K"
    return f"${x:,.0f}"


def kpi_card(label, value, sub=None, negative=False):
    sub_html = ""
    if sub is not None:
        cls = "kpi-sub negative" if negative else "kpi-sub"
        sub_html = f'<div class="{cls}">{sub}</div>'
    st.markdown(
        f'<div class="kpi-card"><div class="kpi-label">{label}</div>'
        f'<div class="kpi-value">{value}</div>{sub_html}</div>',
        unsafe_allow_html=True,
    )


def header(title, subtitle):
    st.markdown(
        f'<div class="main-header"><h1>{title}</h1><p>{subtitle}</p></div>',
        unsafe_allow_html=True,
    )


def section(title):
    st.markdown(f'<div class="section-title">{title}</div>', unsafe_allow_html=True)


# ============================================================================
# PAGE 1 — EXECUTIVE OVERVIEW
# ============================================================================
if page == "Executive Overview":
    header("🤖 Executive Overview", "Corporate AI Adoption — high-level performance snapshot")

    total_companies = df["company_id"].nunique()
    total_investment = df["ai_investment_usd"].sum()
    total_revenue = df["revenue_impact"].sum()
    total_savings = df["cost_savings"].sum()
    avg_roi = df["roi_score"].mean()
    avg_maturity = df["ai_maturity_score"].mean()

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    with c1: kpi_card("Companies", f"{total_companies:,}")
    with c2: kpi_card("Total AI Investment", fmt_usd(total_investment))
    with c3: kpi_card("Total Revenue Impact", fmt_usd(total_revenue))
    with c4: kpi_card("Total Cost Savings", fmt_usd(total_savings))
    with c5: kpi_card("Avg ROI Score", f"{avg_roi:.2f}", negative=avg_roi < 0,
                       sub=("▲ Positive" if avg_roi >= 0 else "▼ Negative"))
    with c6: kpi_card("Avg AI Maturity", f"{avg_maturity:.2f}/10")

    section("📊 Revenue Impact — Country × Industry")
    heatmap_data = df.pivot_table(
        values="revenue_impact", index="country", columns="industry", aggfunc="sum"
    ) / 1e9
    fig = px.imshow(
        heatmap_data, text_auto=".1f", color_continuous_scale="YlGnBu", aspect="auto",
        labels=dict(color="Revenue ($B)"),
    )
    fig.update_layout(template=PLOTLY_TEMPLATE, height=460,
                       xaxis_title="Industry", yaxis_title="Country",
                       margin=dict(l=10, r=10, t=30, b=10))
    st.plotly_chart(fig, width='stretch')

    col_a, col_b = st.columns(2)
    with col_a:
        section("🏆 Top 10 Industries by Avg Revenue Impact")
        top_industries = (
            df.groupby("industry")["revenue_impact"].mean()
            .sort_values(ascending=False).head(10).reset_index()
        )
        fig = px.bar(
            top_industries, x="industry", y="revenue_impact", color="industry",
            color_discrete_sequence=COLOR_SEQ,
            labels={"revenue_impact": "Avg Revenue Impact ($)", "industry": "Industry"},
        )
        fig.update_layout(template=PLOTLY_TEMPLATE, showlegend=False, height=420,
                           xaxis_tickangle=-40, margin=dict(l=10, r=10, t=20, b=10))
        st.plotly_chart(fig, width='stretch')

    with col_b:
        section("📈 AI Adoption Trend Over Time")
        trend = df.groupby("year")["ai_adoption_level"].mean().reset_index()
        fig = px.line(
            trend, x="year", y="ai_adoption_level", markers=True,
            labels={"ai_adoption_level": "Avg Adoption Level", "year": "Year"},
        )
        fig.update_traces(line_color=ACCENT, line_width=3, marker=dict(size=8))
        fig.update_layout(template=PLOTLY_TEMPLATE, height=420, margin=dict(l=10, r=10, t=20, b=10))
        st.plotly_chart(fig, width='stretch')

    # ---- Executive Insights ----
    section("💡 Executive Insights")
    top_ind = df.groupby("industry")["revenue_impact"].mean().idxmax()
    top_ind_val = df.groupby("industry")["revenue_impact"].mean().max()
    top_country = df.groupby("country")["revenue_impact"].sum().idxmax()
    neg_roi_pct = (df["roi_score"] < 0).mean() * 100
    corr = df["ai_investment_usd"].corr(df["revenue_impact"])
    best_year = df.groupby("year")["ai_adoption_level"].mean().idxmax()

    st.markdown(f'<div class="insight-box">🏆 <b>{top_ind}</b> delivers the highest average revenue '
                f'impact at <b>{fmt_usd(top_ind_val)}</b> per company.</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="insight-box">🌍 <b>{top_country}</b> leads in total cumulative revenue '
                f'impact across all filtered companies.</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="insight-box">⚠️ <b>{neg_roi_pct:.1f}%</b> of companies show a negative '
                f'ROI score, indicating investment costs are not yet recovered.</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="insight-box">🔗 The correlation between AI investment and revenue impact '
                f'is <b>{corr:.2f}</b>, suggesting a '
                f'{"strong" if abs(corr) > 0.5 else "moderate" if abs(corr) > 0.2 else "weak"} relationship.</div>',
                unsafe_allow_html=True)


# ============================================================================
# PAGE 2 — FINANCIAL IMPACT
# ============================================================================
elif page == "Financial Impact":
    header("💰 Financial Impact", "Investment returns, efficiency, and business outcomes")

    c1, c2, c3, c4 = st.columns(4)
    with c1: kpi_card("Total AI Investment", fmt_usd(df["ai_investment_usd"].sum()))
    with c2: kpi_card("Total Revenue Impact", fmt_usd(df["revenue_impact"].sum()))
    with c3: kpi_card("Avg Investment Efficiency", f"{df['investment_efficiency'].mean():.2f}")
    with c4: kpi_card("Avg Revenue / AI $", f"{df['revenue_per_ai_dollar'].mean():.2f}")

    section("💵 AI Investment vs. Revenue Impact")
    fig = px.scatter(
        df, x="ai_investment_usd", y="revenue_impact", color="industry",
        opacity=0.6, color_discrete_sequence=COLOR_SEQ,
        hover_data=["company_id", "country"],
        labels={"ai_investment_usd": "AI Investment (USD)", "revenue_impact": "Revenue Impact (USD)"},
    )
    fig.update_layout(template=PLOTLY_TEMPLATE, height=480, margin=dict(l=10, r=10, t=20, b=10))
    st.plotly_chart(fig, width='stretch')

    col_a, col_b = st.columns(2)
    with col_a:
        section("⚙️ Investment Efficiency — Country × Industry")
        eff_heatmap = df.pivot_table(
            values="investment_efficiency", index="country", columns="industry", aggfunc="mean"
        )
        fig = px.imshow(
            eff_heatmap, text_auto=".2f", color_continuous_scale="YlGnBu", aspect="auto",
            labels=dict(color="Efficiency"),
        )
        fig.update_layout(template=PLOTLY_TEMPLATE, height=440,
                           xaxis_title="Industry", yaxis_title="Country",
                           margin=dict(l=10, r=10, t=20, b=10))
        st.plotly_chart(fig, width='stretch')

    with col_b:
        section("📦 Business Impact Score by Industry")
        fig = px.box(
            df, x="industry", y="business_impact_score", color="industry",
            color_discrete_sequence=COLOR_SEQ,
            labels={"business_impact_score": "Business Impact Score ($)"},
        )
        fig.update_layout(template=PLOTLY_TEMPLATE, showlegend=False, height=440,
                           xaxis_tickangle=-40, margin=dict(l=10, r=10, t=20, b=10))
        st.plotly_chart(fig, width='stretch')

    section("📅 Revenue Impact & Cost Savings by Year")
    yearly = df.groupby("year")[["revenue_impact", "cost_savings"]].sum().reset_index()
    fig = go.Figure()
    fig.add_bar(x=yearly["year"], y=yearly["revenue_impact"], name="Revenue Impact", marker_color=ACCENT)
    fig.add_bar(x=yearly["year"], y=yearly["cost_savings"], name="Cost Savings", marker_color="#10b981")
    fig.update_layout(template=PLOTLY_TEMPLATE, barmode="group", height=420,
                       margin=dict(l=10, r=10, t=20, b=10))
    st.plotly_chart(fig, width='stretch')

    section("💡 Financial Insights")
    best_eff_country = df.groupby("country")["investment_efficiency"].mean().idxmax()
    worst_roi_industry = df.groupby("industry")["roi_score"].mean().idxmin()
    median_roi = df["roi_score"].median()
    st.markdown(f'<div class="insight-box">⚙️ <b>{best_eff_country}</b> shows the strongest average '
                f'investment efficiency across industries.</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="insight-box">📉 <b>{worst_roi_industry}</b> has the lowest average ROI '
                f'score, warranting a review of AI spend allocation.</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="insight-box">📊 The median ROI score across all filtered companies is '
                f'<b>{median_roi:.2f}</b>.</div>', unsafe_allow_html=True)


# ============================================================================
# PAGE 3 — OPERATIONAL INSIGHTS
# ============================================================================
elif page == "Operational Insights":
    header("⚙️ Operational Insights", "Automation, productivity, and workforce enablement")

    c1, c2, c3, c4 = st.columns(4)
    with c1: kpi_card("Avg Automation Rate", f"{df['automation_rate'].mean()*100:.1f}%")
    with c2: kpi_card("Avg Productivity Gain", f"{df['productivity_gain'].mean()*100:.1f}%")
    with c3: kpi_card("Avg Training Hours", f"{df['employee_ai_training_hours'].mean():.1f} hrs")
    with c4: kpi_card("Avg Deployments", f"{df['deployment_count'].mean():.1f}")

    col_a, col_b = st.columns(2)
    with col_a:
        section("🚀 Productivity Index by Industry")
        prod = df.groupby("industry")["productivity_index"].mean().sort_values(ascending=False).reset_index()
        fig = px.bar(
            prod, x="productivity_index", y="industry", orientation="h", color="industry",
            color_discrete_sequence=COLOR_SEQ,
            labels={"productivity_index": "Avg Productivity Index", "industry": "Industry"},
        )
        fig.update_layout(template=PLOTLY_TEMPLATE, showlegend=False, height=460,
                           yaxis={"categoryorder": "total ascending"}, margin=dict(l=10, r=10, t=20, b=10))
        st.plotly_chart(fig, width='stretch')

    with col_b:
        section("🔧 Automation Rate vs. Productivity Gain")
        fig = px.scatter(
            df, x="automation_rate", y="productivity_gain", color="industry",
            opacity=0.6, color_discrete_sequence=COLOR_SEQ,
            hover_data=["company_id", "country"],
            labels={"automation_rate": "Automation Rate", "productivity_gain": "Productivity Gain"},
        )
        fig.update_layout(template=PLOTLY_TEMPLATE, height=460, margin=dict(l=10, r=10, t=20, b=10))
        st.plotly_chart(fig, width='stretch')

    col_c, col_d = st.columns(2)
    with col_c:
        section("🎓 Training Hours vs. Productivity Gain")
        fig = px.scatter(
            df, x="employee_ai_training_hours", y="productivity_gain", color="automation_rate",
            color_continuous_scale="Viridis",
            labels={"employee_ai_training_hours": "Training Hours", "productivity_gain": "Productivity Gain"},
        )
        fig.update_layout(template=PLOTLY_TEMPLATE, height=440, margin=dict(l=10, r=10, t=20, b=10))
        st.plotly_chart(fig, width='stretch')

    with col_d:
        section("📦 Deployment Count Distribution")
        fig = px.histogram(
            df, x="deployment_count", nbins=25, color_discrete_sequence=[ACCENT],
            labels={"deployment_count": "Deployment Count"},
        )
        fig.update_layout(template=PLOTLY_TEMPLATE, height=440, margin=dict(l=10, r=10, t=20, b=10))
        st.plotly_chart(fig, width='stretch')

    section("💡 Operational Insights")
    top_auto_industry = df.groupby("industry")["automation_rate"].mean().idxmax()
    train_corr = df["employee_ai_training_hours"].corr(df["productivity_gain"])
    st.markdown(f'<div class="insight-box">🤖 <b>{top_auto_industry}</b> leads in average automation '
                f'rate among the filtered companies.</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="insight-box">🎓 Training hours and productivity gain show a correlation '
                f'of <b>{train_corr:.2f}</b>, '
                f'{"supporting" if train_corr > 0.2 else "showing limited support for"} continued '
                f'investment in employee AI training.</div>', unsafe_allow_html=True)


# ============================================================================
# PAGE 4 — ADVANCED ANALYTICS
# ============================================================================
elif page == "Advanced Analytics":
    header("🔬 Advanced Analytics", "Effectiveness scoring, correlations, and deep-dive metrics")

    c1, c2, c3, c4 = st.columns(4)
    with c1: kpi_card("Avg AI Effectiveness", f"{df['ai_effectiveness_score'].mean():.2f}")
    with c2: kpi_card("Avg ROI Score", f"{df['roi_score'].mean():.2f}")
    with c3: kpi_card("Avg AI Maturity", f"{df['ai_maturity_score'].mean():.2f}/10")
    with c4: kpi_card("Avg Adoption Level", f"{df['ai_adoption_level'].mean()*100:.1f}%")

    section("🔗 Correlation Matrix — Key Metrics")
    numeric_cols = [
        "ai_investment_usd", "revenue_impact", "cost_savings", "automation_rate",
        "productivity_gain", "ai_maturity_score", "ai_adoption_level",
        "roi_score", "investment_efficiency", "ai_effectiveness_score", "business_impact_score",
    ]
    corr_matrix = df[numeric_cols].corr()
    fig = px.imshow(
        corr_matrix, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1, aspect="auto",
    )
    fig.update_layout(template=PLOTLY_TEMPLATE, height=560, margin=dict(l=10, r=10, t=20, b=10))
    st.plotly_chart(fig, width='stretch')

    col_a, col_b = st.columns(2)
    with col_a:
        section("🎯 AI Effectiveness — Country × Industry")
        eff_map = df.pivot_table(
            values="ai_effectiveness_score", index="country", columns="industry", aggfunc="mean"
        )
        fig = px.imshow(
            eff_map, text_auto=".1f", color_continuous_scale="Purples", aspect="auto",
            labels=dict(color="Effectiveness"),
        )
        fig.update_layout(template=PLOTLY_TEMPLATE, height=460,
                           xaxis_title="Industry", yaxis_title="Country", margin=dict(l=10, r=10, t=20, b=10))
        st.plotly_chart(fig, width='stretch')

    with col_b:
        section("📊 ROI Score Distribution")
        fig = px.histogram(
            df, x="roi_score", nbins=30, color_discrete_sequence=[ACCENT],
            labels={"roi_score": "ROI Score"},
        )
        fig.add_vline(x=0, line_dash="dash", line_color="#ef4444")
        fig.update_layout(template=PLOTLY_TEMPLATE, height=460, margin=dict(l=10, r=10, t=20, b=10))
        st.plotly_chart(fig, width='stretch')

    section("🧠 AI Maturity vs. AI Effectiveness Score")
    fig = px.scatter(
        df, x="ai_maturity_score", y="ai_effectiveness_score", color="industry", size="ai_investment_usd",
        opacity=0.6, color_discrete_sequence=COLOR_SEQ, hover_data=["company_id", "country"],
        labels={"ai_maturity_score": "AI Maturity Score", "ai_effectiveness_score": "AI Effectiveness Score"},
    )
    fig.update_layout(template=PLOTLY_TEMPLATE, height=480, margin=dict(l=10, r=10, t=20, b=10))
    st.plotly_chart(fig, width='stretch')

    section("💡 Advanced Insights")
    maturity_corr = df["ai_maturity_score"].corr(df["ai_effectiveness_score"])
    top_effective_industry = df.groupby("industry")["ai_effectiveness_score"].mean().idxmax()
    high_maturity_pct = (df["ai_maturity_score"] >= 7).mean() * 100
    st.markdown(f'<div class="insight-box">🧠 AI maturity score correlates with effectiveness at '
                f'<b>{maturity_corr:.2f}</b>, confirming maturity is a strong effectiveness driver.</div>',
                unsafe_allow_html=True)
    st.markdown(f'<div class="insight-box">🏆 <b>{top_effective_industry}</b> achieves the highest average '
                f'AI effectiveness score among filtered companies.</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="insight-box">📈 <b>{high_maturity_pct:.1f}%</b> of companies have reached '
                f'high AI maturity (score ≥ 7/10).</div>', unsafe_allow_html=True)


# ----------------------------------------------------------------------------
# FOOTER
# ----------------------------------------------------------------------------
st.markdown("---")
st.caption("Corporate AI Adoption Dashboard · Built with Streamlit & Plotly")
