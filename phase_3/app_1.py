"""
AI Adoption & Business Impact Analytics Dashboard
Run with: streamlit run app.py
"""

import os
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

# ----------------------------------------------------------------------------
# PAGE CONFIG
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="AI Adoption & Business Impact Analytics Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

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
    "corporate_ai_adoption_dataset.csv",
    "corporate_dataset.csv",
]

# ----------------------------------------------------------------------------
# DATA LOADING
# ----------------------------------------------------------------------------
@st.cache_data(show_spinner="Loading dataset...")
def load_data(file_source):
    """Load the CSV and validate required columns."""
    df = pd.read_csv(file_source)
    df.columns = [c.strip() for c in df.columns]

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        return None, missing

    # Coerce numeric columns safely, invalid parsing becomes NaN
    for col in NUMERIC_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop rows missing essential identifiers
    df = df.dropna(subset=["industry", "country", "year"])

    # Fill remaining numeric NaNs with column median (safe handling of missing values)
    for col in NUMERIC_COLUMNS:
        if df[col].isnull().any():
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val if not np.isnan(median_val) else 0)

    df["year"] = df["year"].astype(int)
    df["industry"] = df["industry"].astype(str).str.strip()
    df["country"] = df["country"].astype(str).str.strip()

    return df, []


@st.cache_data(show_spinner=False)
def compute_derived_columns(df):
    """Add engineered metrics used across the dashboard."""
    df = df.copy()

    # roi_score = revenue_impact + cost_savings
    df["roi_score"] = df["revenue_impact"] + df["cost_savings"]

    # investment_efficiency = revenue_impact / ai_investment_usd (safe division)
    safe_investment = df["ai_investment_usd"].replace(0, np.nan)
    df["investment_efficiency"] = (df["revenue_impact"] / safe_investment).fillna(0)

    # productivity_index = productivity_gain * automation_rate
    df["productivity_index"] = df["productivity_gain"] * df["automation_rate"]

    # AI Adoption Level bucketed into categories for filtering
    def bucket_adoption(x):
        if pd.isna(x):
            return "Unknown"
        if x < 0.40:
            return "Low"
        elif x < 0.70:
            return "Medium"
        else:
            return "High"

    df["ai_adoption_category"] = df["ai_adoption_level"].apply(bucket_adoption)

    return df


def find_local_dataset():
    for name in CANDIDATE_FILENAMES:
        if os.path.exists(name):
            return name
    return None


# ----------------------------------------------------------------------------
# LOAD DATA (with upload fallback)
# ----------------------------------------------------------------------------
local_path = find_local_dataset()

if local_path is not None:
    raw_df, missing_cols = load_data(local_path)
else:
    st.sidebar.warning("Dataset file not found locally.")
    uploaded_file = st.sidebar.file_uploader("Upload corporate_ai_adoption_dataset.csv", type="csv")
    if uploaded_file is not None:
        raw_df, missing_cols = load_data(uploaded_file)
    else:
        raw_df, missing_cols = None, []

if raw_df is None:
    if missing_cols:
        st.error(
            "The uploaded dataset is missing required columns: "
            + ", ".join(missing_cols)
        )
    else:
        st.title("AI Adoption & Business Impact Analytics Dashboard")
        st.info("Please upload the dataset CSV file using the sidebar to continue.")
    st.stop()

df = compute_derived_columns(raw_df)

# ----------------------------------------------------------------------------
# SIDEBAR - NAVIGATION
# ----------------------------------------------------------------------------
st.sidebar.title("AI Adoption Dashboard")
page = st.sidebar.radio(
    "Navigate",
    [
        "Executive Overview",
        "Financial Impact",
        "Workforce & Productivity",
        "Industry Deep Dive",
        "Predictive Insights",
    ],
)

st.sidebar.markdown("---")
st.sidebar.header("Filters")

# Initialize default filter state
industry_options = sorted(df["industry"].unique().tolist())
country_options = sorted(df["country"].unique().tolist())
year_options = sorted(df["year"].unique().tolist())
adoption_options = ["Low", "Medium", "High"]

defaults = {
    "f_industry": industry_options,
    "f_country": country_options,
    "f_year": year_options,
    "f_adoption": adoption_options,
}
for key, val in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = val

def reset_filters():
    st.session_state["f_industry"] = industry_options
    st.session_state["f_country"] = country_options
    st.session_state["f_year"] = year_options
    st.session_state["f_adoption"] = adoption_options

st.sidebar.multiselect("Industry", industry_options, key="f_industry")
st.sidebar.multiselect("Country", country_options, key="f_country")
st.sidebar.multiselect("Year", year_options, key="f_year")
st.sidebar.multiselect("AI Adoption Level", adoption_options, key="f_adoption")
st.sidebar.button("Reset Filters", on_click=reset_filters, use_container_width=True)

st.sidebar.markdown("---")
st.sidebar.subheader("Dataset Summary")
st.sidebar.metric("Rows", f"{df.shape[0]:,}")
st.sidebar.metric("Columns", f"{df.shape[1]}")
c1, c2 = st.sidebar.columns(2)
c1.metric("Industries", df["industry"].nunique())
c2.metric("Countries", df["country"].nunique())

# ----------------------------------------------------------------------------
# APPLY FILTERS
# ----------------------------------------------------------------------------
def apply_filters(source_df):
    sel_industry = st.session_state["f_industry"] or industry_options
    sel_country = st.session_state["f_country"] or country_options
    sel_year = st.session_state["f_year"] or year_options
    sel_adoption = st.session_state["f_adoption"] or adoption_options

    filtered = source_df[
        source_df["industry"].isin(sel_industry)
        & source_df["country"].isin(sel_country)
        & source_df["year"].isin(sel_year)
        & source_df["ai_adoption_category"].isin(sel_adoption)
    ]
    return filtered


fdf = apply_filters(df)

st.title("AI Adoption & Business Impact Analytics Dashboard")
st.caption("Corporate AI Adoption Dataset — Executive Analytics Suite")

if fdf.empty:
    st.warning("No data available for selected filters.")
    st.stop()

# ----------------------------------------------------------------------------
# HELPER FORMATTERS
# ----------------------------------------------------------------------------
def fmt_currency(value):
    if pd.isna(value):
        return "N/A"
    if abs(value) >= 1_000_000_000:
        return f"${value / 1_000_000_000:,.2f}B"
    if abs(value) >= 1_000_000:
        return f"${value / 1_000_000:,.2f}M"
    if abs(value) >= 1_000:
        return f"${value / 1_000:,.1f}K"
    return f"${value:,.0f}"


def fmt_pct(value):
    if pd.isna(value):
        return "N/A"
    return f"{value * 100:,.1f}%"


PLOTLY_TEMPLATE = "plotly_white"

# ==============================================================================
# PAGE 1: EXECUTIVE OVERVIEW
# ==============================================================================
if page == "Executive Overview":
    st.subheader("Executive Overview")

    k1, k2, k3, k4, k5, k6 = st.columns(6)
    k1.metric("Total Companies", f"{fdf['company_id'].nunique():,}")
    k2.metric("Total AI Investment", fmt_currency(fdf["ai_investment_usd"].sum()))
    k3.metric("Avg Productivity Gain", fmt_pct(fdf["productivity_gain"].mean()))
    k4.metric("Avg Revenue Impact", fmt_currency(fdf["revenue_impact"].mean()))
    k5.metric("Avg AI Maturity Score", f"{fdf['ai_maturity_score'].mean():,.2f}")
    k6.metric("Avg Automation Rate", fmt_pct(fdf["automation_rate"].mean()))

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        industry_adoption = (
            fdf.groupby("industry", as_index=False)["ai_adoption_level"]
            .mean()
            .sort_values("ai_adoption_level", ascending=False)
        )
        fig = px.bar(
            industry_adoption, x="industry", y="ai_adoption_level",
            title="Average AI Adoption Level by Industry",
            labels={"industry": "Industry", "ai_adoption_level": "Avg Adoption Level"},
            template=PLOTLY_TEMPLATE, color="ai_adoption_level",
            color_continuous_scale="Blues",
        )
        fig.update_layout(xaxis_tickangle=-35, coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        country_adoption = (
            fdf.groupby("country", as_index=False)["ai_adoption_level"]
            .mean()
            .sort_values("ai_adoption_level", ascending=False)
        )
        fig = px.bar(
            country_adoption, x="country", y="ai_adoption_level",
            title="Average AI Adoption Level by Country",
            labels={"country": "Country", "ai_adoption_level": "Avg Adoption Level"},
            template=PLOTLY_TEMPLATE, color="ai_adoption_level",
            color_continuous_scale="Teal",
        )
        fig.update_layout(xaxis_tickangle=-35, coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    col3, col4 = st.columns(2)

    with col3:
        yearly_growth = (
            fdf.groupby("year", as_index=False)["ai_adoption_level"]
            .mean()
            .sort_values("year")
        )
        fig = px.line(
            yearly_growth, x="year", y="ai_adoption_level", markers=True,
            title="Year-wise AI Adoption Growth",
            labels={"year": "Year", "ai_adoption_level": "Avg Adoption Level"},
            template=PLOTLY_TEMPLATE,
        )
        st.plotly_chart(fig, use_container_width=True)

    with col4:
        industry_counts = fdf["industry"].value_counts().reset_index()
        industry_counts.columns = ["industry", "count"]
        fig = px.pie(
            industry_counts, names="industry", values="count",
            title="Company Distribution by Industry",
            template=PLOTLY_TEMPLATE, hole=0.45,
        )
        st.plotly_chart(fig, use_container_width=True)

# ==============================================================================
# PAGE 2: FINANCIAL IMPACT
# ==============================================================================
elif page == "Financial Impact":
    st.subheader("Financial Impact")

    col1, col2 = st.columns(2)

    with col1:
        sample_df = fdf.sample(min(5000, len(fdf)), random_state=42)
        fig = px.scatter(
            sample_df, x="ai_investment_usd", y="revenue_impact",
            color="industry", opacity=0.6,
            title="AI Investment vs Revenue Impact",
            labels={"ai_investment_usd": "AI Investment (USD)", "revenue_impact": "Revenue Impact (USD)"},
            template=PLOTLY_TEMPLATE,
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        top_ind_rev = (
            fdf.groupby("industry", as_index=False)["revenue_impact"]
            .mean()
            .sort_values("revenue_impact", ascending=False)
            .head(10)
        )
        fig = px.bar(
            top_ind_rev, x="revenue_impact", y="industry", orientation="h",
            title="Top Industries by Avg Revenue Impact",
            labels={"revenue_impact": "Avg Revenue Impact (USD)", "industry": "Industry"},
            template=PLOTLY_TEMPLATE, color="revenue_impact",
            color_continuous_scale="Greens",
        )
        fig.update_layout(yaxis={"categoryorder": "total ascending"}, coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    col3, col4 = st.columns(2)

    with col3:
        top_country_rev = (
            fdf.groupby("country", as_index=False)["revenue_impact"]
            .mean()
            .sort_values("revenue_impact", ascending=False)
            .head(10)
        )
        fig = px.bar(
            top_country_rev, x="revenue_impact", y="country", orientation="h",
            title="Top Countries by Avg Revenue Impact",
            labels={"revenue_impact": "Avg Revenue Impact (USD)", "country": "Country"},
            template=PLOTLY_TEMPLATE, color="revenue_impact",
            color_continuous_scale="Purples",
        )
        fig.update_layout(yaxis={"categoryorder": "total ascending"}, coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    with col4:
        roi_by_industry = (
            fdf.groupby("industry", as_index=False)["roi_score"]
            .mean()
            .sort_values("roi_score", ascending=False)
        )
        fig = px.bar(
            roi_by_industry, x="industry", y="roi_score",
            title="ROI Analysis — Avg ROI Score by Industry",
            labels={"industry": "Industry", "roi_score": "Avg ROI Score (Revenue Impact + Cost Savings)"},
            template=PLOTLY_TEMPLATE, color="roi_score",
            color_continuous_scale="Oranges",
        )
        fig.update_layout(xaxis_tickangle=-35, coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

# ==============================================================================
# PAGE 3: WORKFORCE & PRODUCTIVITY
# ==============================================================================
elif page == "Workforce & Productivity":
    st.subheader("Workforce & Productivity")

    col1, col2 = st.columns(2)

    with col1:
        sample_df = fdf.sample(min(5000, len(fdf)), random_state=42)
        fig = px.scatter(
            sample_df, x="employee_ai_training_hours", y="ai_maturity_score",
            color="industry", opacity=0.6,
            title="Employee AI Training Hours vs AI Maturity Score",
            labels={"employee_ai_training_hours": "Training Hours", "ai_maturity_score": "AI Maturity Score"},
            template=PLOTLY_TEMPLATE,
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.scatter(
            sample_df, x="automation_rate", y="productivity_gain",
            color="industry", opacity=0.6,
            title="Automation Rate vs Productivity Gain",
            labels={"automation_rate": "Automation Rate", "productivity_gain": "Productivity Gain"},
            template=PLOTLY_TEMPLATE,
        )
        st.plotly_chart(fig, use_container_width=True)

    col3, col4 = st.columns(2)

    with col3:
        fig = px.scatter(
            sample_df, x="ai_adoption_level", y="productivity_gain",
            color="ai_adoption_category", opacity=0.6,
            title="AI Adoption Level vs Productivity Gain",
            labels={"ai_adoption_level": "AI Adoption Level", "productivity_gain": "Productivity Gain"},
            template=PLOTLY_TEMPLATE,
        )
        st.plotly_chart(fig, use_container_width=True)

    with col4:
        fig = px.box(
            fdf, x="industry", y="productivity_gain",
            title="Productivity Gain Distribution by Industry",
            labels={"industry": "Industry", "productivity_gain": "Productivity Gain"},
            template=PLOTLY_TEMPLATE, color="industry",
        )
        fig.update_layout(xaxis_tickangle=-35, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    prod_trend = (
        fdf.groupby("year", as_index=False)["productivity_gain"]
        .mean()
        .sort_values("year")
    )
    fig = px.line(
        prod_trend, x="year", y="productivity_gain", markers=True,
        title="Productivity Gain Trend by Year",
        labels={"year": "Year", "productivity_gain": "Avg Productivity Gain"},
        template=PLOTLY_TEMPLATE,
    )
    st.plotly_chart(fig, use_container_width=True)

# ==============================================================================
# PAGE 4: INDUSTRY DEEP DIVE
# ==============================================================================
elif page == "Industry Deep Dive":
    st.subheader("Industry Deep Dive")

    available_industries = sorted(fdf["industry"].unique().tolist())
    selected_industry = st.selectbox("Select Industry", available_industries)

    idf = fdf[fdf["industry"] == selected_industry]

    if idf.empty:
        st.warning("No data available for selected filters.")
        st.stop()

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("AI Investment", fmt_currency(idf["ai_investment_usd"].sum()))
    k2.metric("Revenue Impact", fmt_currency(idf["revenue_impact"].mean()))
    k3.metric("Cost Savings", fmt_currency(idf["cost_savings"].mean()))
    k4.metric("Productivity Gain", fmt_pct(idf["productivity_gain"].mean()))
    k5.metric("AI Maturity Score", f"{idf['ai_maturity_score'].mean():,.2f}")

    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        fig = px.histogram(
            idf, x="revenue_impact", nbins=40,
            title=f"Revenue Impact Distribution — {selected_industry}",
            labels={"revenue_impact": "Revenue Impact (USD)"},
            template=PLOTLY_TEMPLATE,
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.histogram(
            idf, x="productivity_gain", nbins=40,
            title=f"Productivity Gain Distribution — {selected_industry}",
            labels={"productivity_gain": "Productivity Gain"},
            template=PLOTLY_TEMPLATE,
        )
        st.plotly_chart(fig, use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        inv_trend = idf.groupby("year", as_index=False)["ai_investment_usd"].mean().sort_values("year")
        fig = px.line(
            inv_trend, x="year", y="ai_investment_usd", markers=True,
            title=f"AI Investment Trend — {selected_industry}",
            labels={"year": "Year", "ai_investment_usd": "Avg AI Investment (USD)"},
            template=PLOTLY_TEMPLATE,
        )
        st.plotly_chart(fig, use_container_width=True)

    with col4:
        savings_trend = idf.groupby("year", as_index=False)["cost_savings"].mean().sort_values("year")
        fig = px.line(
            savings_trend, x="year", y="cost_savings", markers=True,
            title=f"Cost Savings Trend — {selected_industry}",
            labels={"year": "Year", "cost_savings": "Avg Cost Savings (USD)"},
            template=PLOTLY_TEMPLATE,
        )
        st.plotly_chart(fig, use_container_width=True)

# ==============================================================================
# PAGE 5: PREDICTIVE INSIGHTS
# ==============================================================================
elif page == "Predictive Insights":
    st.subheader("Predictive Insights")

    engineered_cols = ["roi_score", "investment_efficiency", "productivity_index"]
    st.caption(
        "Engineered metrics: roi_score = revenue_impact + cost_savings · "
        "investment_efficiency = revenue_impact / ai_investment_usd · "
        "productivity_index = productivity_gain × automation_rate"
    )

    corr_cols = [
        "ai_adoption_level", "ai_investment_usd", "automation_rate",
        "cost_savings", "revenue_impact", "productivity_gain",
        "employee_ai_training_hours", "ai_maturity_score",
    ] + engineered_cols

    corr_df = fdf[corr_cols].corr()

    fig = px.imshow(
        corr_df, text_auto=".2f", aspect="auto",
        title="Correlation Matrix",
        color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
        template=PLOTLY_TEMPLATE,
    )
    fig.update_layout(height=600)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.markdown("#### Linear Regression — Predicting AI Maturity Score")

    feature_cols = [
        "employee_ai_training_hours", "ai_investment_usd",
        "automation_rate", "productivity_gain",
    ]
    target_col = "ai_maturity_score"

    reg_df = fdf[feature_cols + [target_col]].dropna()

    if len(reg_df) < 20:
        st.warning("Not enough data available for the selected filters to run a regression model.")
    else:
        X = reg_df[feature_cols]
        y = reg_df[target_col]

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42
        )

        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        r2 = r2_score(y_test, y_pred)

        col1, col2 = st.columns([1, 2])
        with col1:
            st.metric("R² Score (Test Set)", f"{r2:,.3f}")
            st.metric("Training Samples", f"{len(X_train):,}")
            st.metric("Test Samples", f"{len(X_test):,}")

        with col2:
            importance_df = pd.DataFrame({
                "feature": feature_cols,
                "coefficient": model.coef_,
            })
            importance_df["abs_coefficient"] = importance_df["coefficient"].abs()
            importance_df = importance_df.sort_values("abs_coefficient", ascending=False)

            fig = px.bar(
                importance_df, x="abs_coefficient", y="feature", orientation="h",
                title="Feature Importance — Key Factors Influencing AI Maturity Score",
                labels={"abs_coefficient": "Standardized Coefficient Magnitude", "feature": "Feature"},
                template=PLOTLY_TEMPLATE, color="coefficient",
                color_continuous_scale="RdBu_r",
            )
            fig.update_layout(yaxis={"categoryorder": "total ascending"})
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("##### Key Factors Influencing AI Success")
        top_factor = importance_df.iloc[0]
        direction = "positively" if top_factor["coefficient"] > 0 else "negatively"
        st.info(
            f"**{top_factor['feature']}** has the strongest association with AI maturity score, "
            f"affecting it {direction} among the selected features. "
            f"The model explains **{r2 * 100:.1f}%** of the variance in AI maturity score "
            f"on the held-out test data for the currently selected filters."
        )
