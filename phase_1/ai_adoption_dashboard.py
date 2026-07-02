"""
Corporate AI Adoption Dashboard
--------------------------------
A professional Streamlit dashboard for exploring the corporate AI adoption
dataset. Run with:

    streamlit run ai_adoption_dashboard.py

Expects a CSV file named `corporate_ai_adoption_dataset.csv` in the same
directory (or upload it via the sidebar).
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# --------------------------------------------------------------------------
# Page configuration
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="Corporate AI Adoption Dashboard",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------------------------------
# Styling
# --------------------------------------------------------------------------
CUSTOM_CSS = """
<style>
    /* Overall page */
    .main {
        background-color: #f5f7fa;
    }

    /* KPI card */
    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, #ffffff 0%, #f8f9fc 100%);
        border: 1px solid #e6e9f0;
        border-radius: 14px;
        padding: 18px 20px 14px 20px;
        box-shadow: 0 2px 10px rgba(15, 23, 42, 0.05);
    }
    div[data-testid="stMetric"] label {
        color: #64748b;
        font-weight: 600;
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }
    div[data-testid="stMetricValue"] {
        color: #1e293b;
        font-weight: 700;
    }

    /* Section headers */
    .section-header {
        font-size: 1.15rem;
        font-weight: 700;
        color: #1e293b;
        margin-top: 0.5rem;
        margin-bottom: 0.75rem;
        border-left: 5px solid #4f46e5;
        padding-left: 10px;
    }

    /* Title block */
    .dashboard-title {
        font-size: 2.1rem;
        font-weight: 800;
        color: #1e293b;
        margin-bottom: 0;
    }
    .dashboard-subtitle {
        color: #64748b;
        font-size: 1rem;
        margin-top: 0;
        margin-bottom: 1.2rem;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #111827;
    }
    section[data-testid="stSidebar"] * {
        color: #e5e7eb !important;
    }

    hr {
        border: none;
        border-top: 1px solid #e2e8f0;
        margin: 1.2rem 0;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# --------------------------------------------------------------------------
# Data loading
# --------------------------------------------------------------------------
DATA_PATH = "corporate_ai_adoption_dataset.csv"

COLOR_SEQUENCE = px.colors.qualitative.Bold
PRIMARY_COLOR = "#4f46e5"


@st.cache_data(show_spinner="Loading dataset...")
def load_data(file) -> pd.DataFrame:
    df = pd.read_csv(file)
    return df


st.sidebar.title("🤖 AI Adoption Dashboard")
st.sidebar.markdown("Use the filters below to explore the dataset.")

uploaded_file = st.sidebar.file_uploader(
    "Upload a CSV (optional)", type=["csv"], help="Defaults to the bundled dataset if none is uploaded."
)

try:
    if uploaded_file is not None:
        df = load_data(uploaded_file)
    else:
        df = load_data(DATA_PATH)
except FileNotFoundError:
    st.error(
        f"Could not find `{DATA_PATH}`. Please place the dataset in the same "
        "folder as this script, or upload a CSV using the sidebar."
    )
    st.stop()

# --------------------------------------------------------------------------
# Sidebar filters
# --------------------------------------------------------------------------
st.sidebar.markdown("---")
st.sidebar.subheader("Filters")

year_min, year_max = int(df["year"].min()), int(df["year"].max())
year_range = st.sidebar.slider(
    "Year range",
    min_value=year_min,
    max_value=year_max,
    value=(year_min, year_max),
    step=1,
)

industries = sorted(df["industry"].unique().tolist())
selected_industries = st.sidebar.multiselect(
    "Industry", options=industries, default=industries
)

countries = sorted(df["country"].unique().tolist())
selected_countries = st.sidebar.multiselect(
    "Country", options=countries, default=countries
)

st.sidebar.markdown("---")
if st.sidebar.button("Reset filters"):
    st.rerun()

# --------------------------------------------------------------------------
# Apply filters
# --------------------------------------------------------------------------
mask = (
    df["year"].between(year_range[0], year_range[1])
    & df["industry"].isin(selected_industries if selected_industries else industries)
    & df["country"].isin(selected_countries if selected_countries else countries)
)
fdf = df.loc[mask].copy()

if fdf.empty:
    st.warning("No data matches the selected filters. Please broaden your selection.")
    st.stop()

# --------------------------------------------------------------------------
# Header
# --------------------------------------------------------------------------
st.markdown('<p class="dashboard-title">Corporate AI Adoption Dashboard</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="dashboard-subtitle">Investment, productivity, and maturity trends across '
    'industries and countries</p>',
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------------
# KPI Cards
# --------------------------------------------------------------------------
total_companies = fdf["company_id"].nunique()
total_investment = fdf["ai_investment_usd"].sum()
avg_productivity_gain = fdf["productivity_gain"].mean()
avg_revenue_growth = fdf["revenue_impact"].mean()
avg_maturity_score = fdf["ai_maturity_score"].mean()
avg_automation_rate = fdf["automation_rate"].mean()


def format_currency(value: float) -> str:
    if abs(value) >= 1e9:
        return f"${value / 1e9:,.2f}B"
    if abs(value) >= 1e6:
        return f"${value / 1e6:,.2f}M"
    if abs(value) >= 1e3:
        return f"${value / 1e3:,.1f}K"
    return f"${value:,.0f}"


kpi_cols = st.columns(6)

kpi_cols[0].metric("Total Companies", f"{total_companies:,}")
kpi_cols[1].metric("Total AI Investment", format_currency(total_investment))
kpi_cols[2].metric("Avg Productivity Gain", f"{avg_productivity_gain * 100:.1f}%")
kpi_cols[3].metric("Avg Revenue Impact", format_currency(avg_revenue_growth))
kpi_cols[4].metric("Avg AI Maturity Score", f"{avg_maturity_score:.2f} / 10")
kpi_cols[5].metric("Avg Automation Rate", f"{avg_automation_rate * 100:.1f}%")

st.markdown("<hr>", unsafe_allow_html=True)

# --------------------------------------------------------------------------
# Row 1: AI Adoption by Industry & by Country
# --------------------------------------------------------------------------
row1_col1, row1_col2 = st.columns(2)

with row1_col1:
    st.markdown('<p class="section-header">AI Adoption by Industry</p>', unsafe_allow_html=True)
    industry_agg = (
        fdf.groupby("industry", as_index=False)["ai_adoption_level"]
        .mean()
        .sort_values("ai_adoption_level", ascending=True)
    )
    fig_industry = px.bar(
        industry_agg,
        x="ai_adoption_level",
        y="industry",
        orientation="h",
        text=industry_agg["ai_adoption_level"].map(lambda v: f"{v * 100:.1f}%"),
        color="ai_adoption_level",
        color_continuous_scale="Purples",
    )
    fig_industry.update_traces(textposition="outside", cliponaxis=False)
    fig_industry.update_layout(
        xaxis_title="Average AI Adoption Level",
        yaxis_title="",
        xaxis_tickformat=".0%",
        coloraxis_showscale=False,
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(l=10, r=10, t=10, b=10),
        height=420,
    )
    st.plotly_chart(fig_industry, use_container_width=True)

with row1_col2:
    st.markdown('<p class="section-header">AI Adoption by Country</p>', unsafe_allow_html=True)
    country_agg = (
        fdf.groupby("country", as_index=False)["ai_adoption_level"]
        .mean()
        .sort_values("ai_adoption_level", ascending=True)
    )
    fig_country = px.bar(
        country_agg,
        x="ai_adoption_level",
        y="country",
        orientation="h",
        text=country_agg["ai_adoption_level"].map(lambda v: f"{v * 100:.1f}%"),
        color="ai_adoption_level",
        color_continuous_scale="Teal",
    )
    fig_country.update_traces(textposition="outside", cliponaxis=False)
    fig_country.update_layout(
        xaxis_title="Average AI Adoption Level",
        yaxis_title="",
        xaxis_tickformat=".0%",
        coloraxis_showscale=False,
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(l=10, r=10, t=10, b=10),
        height=420,
    )
    st.plotly_chart(fig_country, use_container_width=True)

st.markdown("<hr>", unsafe_allow_html=True)

# --------------------------------------------------------------------------
# Row 2: Year-wise AI Growth
# --------------------------------------------------------------------------
st.markdown('<p class="section-header">Year-wise AI Growth</p>', unsafe_allow_html=True)

yearly_agg = (
    fdf.groupby("year", as_index=False)
    .agg(
        avg_adoption=("ai_adoption_level", "mean"),
        avg_maturity=("ai_maturity_score", "mean"),
        total_investment=("ai_investment_usd", "sum"),
    )
    .sort_values("year")
)

metric_choice = st.radio(
    "Metric",
    options=["Avg AI Adoption Level", "Avg AI Maturity Score", "Total AI Investment"],
    horizontal=True,
    label_visibility="collapsed",
)

fig_year = go.Figure()

if metric_choice == "Avg AI Adoption Level":
    fig_year.add_trace(
        go.Scatter(
            x=yearly_agg["year"],
            y=yearly_agg["avg_adoption"],
            mode="lines+markers",
            line=dict(color=PRIMARY_COLOR, width=3),
            marker=dict(size=8),
            fill="tozeroy",
            fillcolor="rgba(79, 70, 229, 0.10)",
            name="Avg AI Adoption Level",
        )
    )
    fig_year.update_layout(yaxis_tickformat=".0%", yaxis_title="Avg AI Adoption Level")
elif metric_choice == "Avg AI Maturity Score":
    fig_year.add_trace(
        go.Scatter(
            x=yearly_agg["year"],
            y=yearly_agg["avg_maturity"],
            mode="lines+markers",
            line=dict(color="#0d9488", width=3),
            marker=dict(size=8),
            fill="tozeroy",
            fillcolor="rgba(13, 148, 136, 0.10)",
            name="Avg AI Maturity Score",
        )
    )
    fig_year.update_layout(yaxis_title="Avg AI Maturity Score (0-10)")
else:
    fig_year.add_trace(
        go.Scatter(
            x=yearly_agg["year"],
            y=yearly_agg["total_investment"],
            mode="lines+markers",
            line=dict(color="#c026d3", width=3),
            marker=dict(size=8),
            fill="tozeroy",
            fillcolor="rgba(192, 38, 211, 0.10)",
            name="Total AI Investment",
        )
    )
    fig_year.update_layout(yaxis_title="Total AI Investment (USD)")

fig_year.update_layout(
    xaxis_title="Year",
    plot_bgcolor="white",
    paper_bgcolor="white",
    margin=dict(l=10, r=10, t=20, b=10),
    height=420,
    hovermode="x unified",
)
st.plotly_chart(fig_year, use_container_width=True)

st.markdown("<hr>", unsafe_allow_html=True)

# --------------------------------------------------------------------------
# Row 3: Additional insight — Industry x Country heatmap and raw data
# --------------------------------------------------------------------------
row3_col1, row3_col2 = st.columns([1.3, 1])

with row3_col1:
    st.markdown('<p class="section-header">Adoption Heatmap: Industry vs Country</p>', unsafe_allow_html=True)
    heat_data = (
        fdf.groupby(["industry", "country"], as_index=False)["ai_adoption_level"].mean()
    )
    heat_pivot = heat_data.pivot(index="industry", columns="country", values="ai_adoption_level")
    fig_heat = px.imshow(
        heat_pivot,
        color_continuous_scale="Purples",
        aspect="auto",
        labels=dict(color="Avg Adoption"),
    )
    fig_heat.update_layout(
        margin=dict(l=10, r=10, t=10, b=10),
        height=420,
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    st.plotly_chart(fig_heat, use_container_width=True)

with row3_col2:
    st.markdown('<p class="section-header">Automation Rate Distribution</p>', unsafe_allow_html=True)
    fig_hist = px.histogram(
        fdf,
        x="automation_rate",
        nbins=30,
        color_discrete_sequence=[PRIMARY_COLOR],
    )
    fig_hist.update_layout(
        xaxis_title="Automation Rate",
        xaxis_tickformat=".0%",
        yaxis_title="Number of Companies",
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(l=10, r=10, t=10, b=10),
        height=420,
        bargap=0.05,
    )
    st.plotly_chart(fig_hist, use_container_width=True)

st.markdown("<hr>", unsafe_allow_html=True)

# --------------------------------------------------------------------------
# Data table
# --------------------------------------------------------------------------
with st.expander("View filtered raw data"):
    st.dataframe(fdf, use_container_width=True, height=400)
    csv_bytes = fdf.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download filtered data as CSV",
        data=csv_bytes,
        file_name="filtered_ai_adoption_data.csv",
        mime="text/csv",
    )

st.markdown(
    '<p style="text-align:center; color:#94a3b8; font-size:0.8rem; margin-top:1.5rem;">'
    "Corporate AI Adoption Dashboard · Built with Streamlit &amp; Plotly</p>",
    unsafe_allow_html=True,
)
