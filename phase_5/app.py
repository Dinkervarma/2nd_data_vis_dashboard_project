"""
app.py
------
AI Adoption & Corporate Performance Dashboard
A Streamlit port of the Phase 5 analysis notebook, redesigned as an
interactive, dark-themed, Power BI-style dashboard.

Run locally with:
    streamlit run app.py

The dataset (corporate_dataset.csv) ships alongside this app and is loaded
automatically and cached — there is no upload step required.
"""

import os
import io
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from data_utils import (
    load_data,
    apply_filters,
    compute_kpis,
    compute_correlation,
    generate_insights,
)

# ---------------------------------------------------------------------------
# Page config (must be the first Streamlit call)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="AI Adoption Corporate Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Load custom CSS
# ---------------------------------------------------------------------------
def load_css(path: str):
    if os.path.exists(path):
        with open(path) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css(os.path.join(os.path.dirname(__file__), "assets", "style.css"))

# Plotly template shared by every chart for visual consistency
PLOTLY_TEMPLATE = "plotly_dark"
ACCENT = "#3B82F6"
ACCENT_SOFT = "#60A5FA"
COLOR_SEQUENCE = ["#3B82F6", "#60A5FA", "#22C55E", "#F59E0B", "#A78BFA", "#EF4444", "#14B8A6", "#EC4899", "#84CC16", "#F97316"]

def style_fig(fig, height=420):
    """Apply consistent dark styling to every Plotly figure."""
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#D1D5DB", size=12),
        margin=dict(l=10, r=10, t=50, b=10),
        height=height,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


# ---------------------------------------------------------------------------
# Data loading (cached — CSV is bundled, never uploaded by the user)
# ---------------------------------------------------------------------------
try:
    df_raw = load_data()
except FileNotFoundError:
    st.error(
        "Could not find `corporate_dataset.csv` next to app.py. "
        "Make sure the CSV is deployed in the same folder as this script."
    )
    st.stop()

if df_raw.empty:
    st.error("The dataset loaded but contains no rows. Please check the source file.")
    st.stop()

# ---------------------------------------------------------------------------
# Sidebar — filters
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🎛️ Dashboard Filters")
    st.caption("Filters apply instantly to every chart and KPI below.")

    industries = st.multiselect(
        "Industry",
        options=sorted(df_raw["industry"].unique()),
        default=[],
        placeholder="All industries",
    )

    regions = st.multiselect(
        "Region",
        options=sorted(df_raw["region"].unique()),
        default=[],
        placeholder="All regions",
        help="Derived from company country (continent-level grouping).",
    )

    sizes = st.multiselect(
        "Company Size",
        options=["Small", "Mid-size", "Large", "Enterprise"],
        default=[],
        placeholder="All sizes",
        help="Derived from AI investment quartiles (proxy for company scale).",
    )

    year_min, year_max = int(df_raw["year"].min()), int(df_raw["year"].max())
    years = st.slider(
        "Year range",
        min_value=year_min,
        max_value=year_max,
        value=(year_min, year_max),
    )

    adoption_bands = st.multiselect(
        "AI Adoption Level",
        options=["Low (0-25%)", "Moderate (25-50%)", "High (50-75%)", "Very High (75-100%)"],
        default=[],
        placeholder="All adoption levels",
    )

    st.divider()
    if st.button("🔄 Reset all filters", use_container_width=True):
        st.rerun()

    st.divider()
    st.caption(f"Dataset: **{len(df_raw):,}** records · {df_raw['company_id'].nunique():,} companies")
    st.caption("Source: corporate_dataset.csv (bundled — no upload needed)")

# ---------------------------------------------------------------------------
# Apply filters
# ---------------------------------------------------------------------------
df = apply_filters(df_raw, industries, regions, sizes, years, adoption_bands)

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown(
    f"""
    <div class="dashboard-header">
        <div>
            <h1>📊 AI Adoption &amp; Corporate Performance Dashboard</h1>
            <p>Enterprise AI investment, adoption, and impact analytics — {len(df):,} of {len(df_raw):,} records in view</p>
        </div>
        <div class="badge">Live · Phase 5 Analysis</div>
    </div>
    """,
    unsafe_allow_html=True,
)

if df.empty:
    st.warning("No records match the current filter selection. Try widening your filters in the sidebar.")
    st.stop()

# ---------------------------------------------------------------------------
# KPI Cards
# ---------------------------------------------------------------------------
kpis = compute_kpis(df)

def kpi_card(label, value, sub=None, sub_class=""):
    sub_html = f'<div class="kpi-sub {sub_class}">{sub}</div>' if sub else ""
    return f"""
    <div class="kpi-card">
        <div class="kpi-label">{label}</div>
        <div class="kpi-value">{value}</div>
        {sub_html}
    </div>
    """

col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    st.markdown(kpi_card("Avg AI Adoption", f"{kpis['ai_adoption_level']:.1%}",
                          f"{kpis['company_count']:,} companies", "positive"), unsafe_allow_html=True)
with col2:
    st.markdown(kpi_card("Avg Productivity Gain", f"{kpis['productivity_gain']:.1%}",
                          "vs. pre-AI baseline"), unsafe_allow_html=True)
with col3:
    st.markdown(kpi_card("Avg Revenue Impact", f"${kpis['revenue_impact']:,.0f}",
                          "per company"), unsafe_allow_html=True)
with col4:
    st.markdown(kpi_card("Avg AI Investment", f"${kpis['ai_investment_usd']:,.0f}",
                          f"${kpis['total_cost_savings']:,.0f} total saved"), unsafe_allow_html=True)
with col5:
    st.markdown(kpi_card("Avg Training Hours", f"{kpis['employee_ai_training_hours']:.1f} hrs",
                          f"Maturity score {kpis['avg_maturity_score']:.1f}/10"), unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Row 1: Adoption trend + Industry breakdown
# ---------------------------------------------------------------------------
st.markdown('<div class="section-header"><div class="dot"></div><h3>Trends &amp; Industry Breakdown</h3></div>', unsafe_allow_html=True)

r1c1, r1c2 = st.columns([1.3, 1])

with r1c1:
    trend = (
        df.groupby("year")
        .agg(
            avg_adoption=("ai_adoption_level", "mean"),
            avg_investment=("ai_investment_usd", "mean"),
        )
        .reset_index()
    )
    fig_trend = go.Figure()
    fig_trend.add_trace(go.Scatter(
        x=trend["year"], y=trend["avg_adoption"], name="Avg Adoption Level",
        mode="lines+markers", line=dict(color=ACCENT, width=3), yaxis="y1",
    ))
    fig_trend.add_trace(go.Bar(
        x=trend["year"], y=trend["avg_investment"], name="Avg AI Investment (USD)",
        marker_color="rgba(96,165,250,0.35)", yaxis="y2",
    ))
    fig_trend.update_layout(
        title="AI Adoption &amp; Investment Trend Over Time",
        yaxis=dict(title="Avg Adoption Level", tickformat=".0%"),
        yaxis2=dict(title="Avg Investment (USD)", overlaying="y", side="right", showgrid=False),
        xaxis=dict(title="Year", dtick=1),
        barmode="overlay",
    )
    st.plotly_chart(style_fig(fig_trend), use_container_width=True)

with r1c2:
    ind = df.groupby("industry")["ai_adoption_level"].mean().sort_values(ascending=True).reset_index()
    fig_ind = px.bar(
        ind, x="ai_adoption_level", y="industry", orientation="h",
        color="ai_adoption_level", color_continuous_scale="Blues",
        title="Avg AI Adoption by Industry",
        labels={"ai_adoption_level": "Avg Adoption Level", "industry": ""},
    )
    fig_ind.update_layout(xaxis_tickformat=".0%", coloraxis_showscale=False)
    st.plotly_chart(style_fig(fig_ind), use_container_width=True)

# ---------------------------------------------------------------------------
# Row 2: Region distribution + Investment vs Revenue scatter
# ---------------------------------------------------------------------------
r2c1, r2c2 = st.columns([1, 1.3])

with r2c1:
    region_counts = df.groupby("region")["company_id"].nunique().reset_index(name="companies")
    fig_region = px.pie(
        region_counts, names="region", values="companies", hole=0.55,
        color_discrete_sequence=COLOR_SEQUENCE,
        title="Company Distribution by Region",
    )
    fig_region.update_traces(textinfo="percent+label", textfont_size=11)
    st.plotly_chart(style_fig(fig_region), use_container_width=True)

with r2c2:
    sample = df.sample(min(3000, len(df)), random_state=42)  # keep scatter responsive on 200k rows
    fig_scatter = px.scatter(
        sample, x="ai_investment_usd", y="revenue_impact", color="industry",
        size="deployment_count", size_max=18, opacity=0.7,
        color_discrete_sequence=COLOR_SEQUENCE,
        title="AI Investment vs Revenue Impact (sampled for responsiveness)",
        labels={"ai_investment_usd": "AI Investment (USD)", "revenue_impact": "Revenue Impact (USD)"},
        hover_data=["company_id", "country", "year"],
    )
    st.plotly_chart(style_fig(fig_scatter), use_container_width=True)

# ---------------------------------------------------------------------------
# Row 3: Correlation Heatmap (interactive — hover, zoom, download)
# ---------------------------------------------------------------------------
st.markdown('<div class="section-header"><div class="dot"></div><h3>Correlation Analysis</h3></div>', unsafe_allow_html=True)

corr = compute_correlation(df)

fig_corr = px.imshow(
    corr,
    text_auto=".2f",
    color_continuous_scale="RdBu_r",
    zmin=-1, zmax=1,
    aspect="auto",
    labels=dict(color="Correlation"),
)
fig_corr.update_xaxes(tickangle=45)
fig_corr.update_layout(title="Correlation Matrix — Key AI &amp; Performance Metrics")
fig_corr = style_fig(fig_corr, height=550)
fig_corr.update_layout(
    modebar_add=["zoom", "pan", "resetScale2d", "toImage"],
)

st.plotly_chart(
    fig_corr,
    use_container_width=True,
    config={
        "displaylogo": False,
        "modeBarButtonsToAdd": ["zoom2d", "pan2d", "resetScale2d"],
        "toImageButtonOptions": {"format": "png", "filename": "correlation_heatmap", "scale": 2},
    },
)

with st.expander("📥 Download correlation matrix as CSV"):
    csv_buffer = io.StringIO()
    corr.to_csv(csv_buffer)
    st.download_button(
        label="Download correlation_matrix.csv",
        data=csv_buffer.getvalue(),
        file_name="correlation_matrix.csv",
        mime="text/csv",
    )

st.caption(
    "💡 Reading the heatmap: values near **+1** indicate a strong positive relationship, "
    "values near **-1** indicate a strong inverse relationship, and values near **0** indicate "
    "little to no linear relationship. Hover any cell for the exact coefficient, drag to zoom, "
    "or use the camera icon to export as an image."
)

# ---------------------------------------------------------------------------
# Row 4: Business Insights
# ---------------------------------------------------------------------------
st.markdown('<div class="section-header"><div class="dot"></div><h3>Business Insights</h3></div>', unsafe_allow_html=True)

insights = generate_insights(df, corr)
ins_col1, ins_col2 = st.columns(2)
for i, insight in enumerate(insights):
    target_col = ins_col1 if i % 2 == 0 else ins_col2
    with target_col:
        st.markdown(f'<div class="insight-card">{insight}</div>', unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Row 5: Detail table (optional, collapsed)
# ---------------------------------------------------------------------------
with st.expander("🔎 View filtered data table"):
    st.dataframe(
        df[[
            "company_id", "industry", "country", "region", "company_size", "year",
            "ai_adoption_level", "ai_investment_usd", "automation_rate",
            "cost_savings", "revenue_impact", "productivity_gain",
            "employee_ai_training_hours", "ai_maturity_score", "deployment_count",
        ]],
        use_container_width=True,
        height=350,
    )
    csv_out = df.to_csv(index=False).encode("utf-8")
    st.download_button("Download filtered data as CSV", data=csv_out,
                        file_name="filtered_corporate_data.csv", mime="text/csv")

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div class="dashboard-footer">
        Built with Streamlit &amp; Plotly · AI Adoption &amp; Corporate Performance Dashboard · Phase 5
    </div>
    """,
    unsafe_allow_html=True,
)
