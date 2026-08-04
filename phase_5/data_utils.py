"""
data_utils.py
--------------
Data loading, cleaning, and derived-feature logic for the AI Adoption
Corporate Dashboard. All heavy computation lives here and is cached so the
Streamlit app stays fast on repeated interactions.

NOTE ON DERIVED COLUMNS
The source CSV (corporate_dataset.csv) does NOT contain "Region" or
"Company Size" columns. Since the dashboard spec calls for filters on both,
we derive them here from existing columns:
  - Region        <- mapped from `country` (continent-level grouping)
  - Company Size  <- bucketed from `ai_investment_usd` (quartiles),
                      used as a practical proxy for company scale
These are clearly labeled as derived/estimated in the UI.
"""

import os
import pandas as pd
import numpy as np
import streamlit as st

# ---------------------------------------------------------------------------
# Path to the bundled dataset. The CSV ships alongside the app so the user
# never has to upload a file — it is read directly from disk every run
# (but only re-parsed when the file changes, thanks to st.cache_data).
# ---------------------------------------------------------------------------
DATA_PATH = os.path.join(os.path.dirname(__file__), "corporate_dataset.csv")

# Country -> Region mapping (derived, not in original data)
COUNTRY_TO_REGION = {
    "United States": "North America",
    "Canada": "North America",
    "Brazil": "South America",
    "United Kingdom": "Europe",
    "Germany": "Europe",
    "France": "Europe",
    "Netherlands": "Europe",
    "Sweden": "Europe",
    "China": "Asia Pacific",
    "Japan": "Asia Pacific",
    "India": "Asia Pacific",
    "Singapore": "Asia Pacific",
    "South Korea": "Asia Pacific",
    "Australia": "Asia Pacific",
    "UAE": "Middle East",
}

CORR_COLUMNS = [
    "ai_adoption_level",
    "ai_investment_usd",
    "automation_rate",
    "cost_savings",
    "revenue_impact",
    "productivity_gain",
    "employee_ai_training_hours",
    "ai_maturity_score",
    "deployment_count",
]

KPI_TARGET_COLUMNS = [
    "ai_adoption_level",
    "productivity_gain",
    "revenue_impact",
    "ai_investment_usd",
    "employee_ai_training_hours",
]


@st.cache_data(show_spinner="Loading corporate dataset...")
def load_data(path: str = DATA_PATH) -> pd.DataFrame:
    """Load the CSV once and cache it for the whole session."""
    df = pd.read_csv(path)

    # Basic cleanup: strip whitespace on string columns, drop exact dupes
    for col in ["industry", "country", "company_id"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
    df = df.drop_duplicates()

    # --- Derived: Region (from country) ---
    df["region"] = df["country"].map(COUNTRY_TO_REGION).fillna("Other")

    # --- Derived: Company Size (from AI investment quartiles) ---
    # Practical proxy since no headcount/revenue column exists in the source data.
    try:
        df["company_size"] = pd.qcut(
            df["ai_investment_usd"],
            q=4,
            labels=["Small", "Mid-size", "Large", "Enterprise"],
        )
    except ValueError:
        df["company_size"] = "Unclassified"

    # --- Derived: AI Adoption Level bucket (for categorical filtering) ---
    df["adoption_band"] = pd.cut(
        df["ai_adoption_level"],
        bins=[-0.01, 0.25, 0.5, 0.75, 1.0],
        labels=["Low (0-25%)", "Moderate (25-50%)", "High (50-75%)", "Very High (75-100%)"],
    )

    return df


def apply_filters(
    df: pd.DataFrame,
    industries,
    regions,
    sizes,
    years,
    adoption_bands,
) -> pd.DataFrame:
    """Apply sidebar filter selections to the dataframe."""
    filtered = df.copy()

    if industries:
        filtered = filtered[filtered["industry"].isin(industries)]
    if regions:
        filtered = filtered[filtered["region"].isin(regions)]
    if sizes:
        filtered = filtered[filtered["company_size"].astype(str).isin(sizes)]
    if years:
        filtered = filtered[filtered["year"].between(years[0], years[1])]
    if adoption_bands:
        filtered = filtered[filtered["adoption_band"].astype(str).isin(adoption_bands)]

    return filtered


def compute_kpis(df: pd.DataFrame) -> dict:
    """
    Compute the core KPI averages — mirrors the notebook's Cell 5 logic
    (averages of ai_adoption_level, productivity_gain, revenue_impact,
    ai_investment_usd, employee_ai_training_hours), extended with a few
    extra portfolio-friendly metrics (totals, counts).
    """
    if df.empty:
        return {col: 0 for col in KPI_TARGET_COLUMNS} | {
            "company_count": 0,
            "total_cost_savings": 0,
            "avg_automation_rate": 0,
            "avg_maturity_score": 0,
        }

    averages = df[KPI_TARGET_COLUMNS].mean().round(4)

    return {
        "ai_adoption_level": averages["ai_adoption_level"],
        "productivity_gain": averages["productivity_gain"],
        "revenue_impact": averages["revenue_impact"],
        "ai_investment_usd": averages["ai_investment_usd"],
        "employee_ai_training_hours": averages["employee_ai_training_hours"],
        "company_count": df["company_id"].nunique(),
        "total_cost_savings": df["cost_savings"].sum(),
        "avg_automation_rate": df["automation_rate"].mean(),
        "avg_maturity_score": df["ai_maturity_score"].mean(),
    }


def compute_correlation(df: pd.DataFrame) -> pd.DataFrame:
    """
    Correlation matrix over the same numeric columns used in the notebook's
    heatmap (Cell 4), now feeding an interactive Plotly heatmap instead of
    a static seaborn/matplotlib image.
    """
    cols = [c for c in CORR_COLUMNS if c in df.columns]
    if df.empty or len(df) < 2:
        return pd.DataFrame(np.nan, index=cols, columns=cols)
    return df[cols].corr().round(3)


def generate_insights(df: pd.DataFrame, corr: pd.DataFrame) -> list:
    """
    Turn the numbers into a handful of plain-English business insights.
    Rule-based (not a black box) so results stay reproducible and explainable.
    """
    insights = []
    if df.empty:
        return ["No data matches the current filter selection. Try widening your filters."]

    # 1. Top industry by average adoption
    by_industry = df.groupby("industry")["ai_adoption_level"].mean().sort_values(ascending=False)
    if not by_industry.empty:
        top_ind, top_val = by_industry.index[0], by_industry.iloc[0]
        insights.append(
            f"**{top_ind}** leads AI adoption among the filtered companies, "
            f"averaging **{top_val:.1%}** adoption level."
        )

    # 2. Strongest correlation pair (excluding self-correlation)
    if not corr.empty and corr.notna().values.any():
        corr_unstacked = corr.where(~np.eye(len(corr), dtype=bool)).unstack().dropna()
        if not corr_unstacked.empty:
            strongest = corr_unstacked.abs().idxmax()
            strongest_val = corr.loc[strongest[0], strongest[1]]
            direction = "positive" if strongest_val > 0 else "negative"
            insights.append(
                f"**{strongest[0].replace('_', ' ').title()}** and "
                f"**{strongest[1].replace('_', ' ').title()}** show the strongest "
                f"{direction} relationship (r = {strongest_val:.2f})."
            )

    # 3. Region with highest ROI proxy (revenue_impact / ai_investment_usd)
    # Vectorized groupby (avoids pandas-version-specific .apply kwargs).
    region_sums = df.groupby("region")[["revenue_impact", "ai_investment_usd"]].sum()
    region_sums = region_sums[region_sums["ai_investment_usd"] > 0]
    region_roi = (region_sums["revenue_impact"] / region_sums["ai_investment_usd"]).sort_values(ascending=False)
    if not region_roi.empty:
        top_region = region_roi.index[0]
        insights.append(
            f"**{top_region}** delivers the best return on AI investment, generating "
            f"**${region_roi.iloc[0]:.2f}** in revenue impact per $1 invested."
        )

    # 4. Training hours vs productivity relationship
    if "employee_ai_training_hours" in corr.index and "productivity_gain" in corr.columns:
        train_prod_corr = corr.loc["employee_ai_training_hours", "productivity_gain"]
        if pd.notna(train_prod_corr):
            strength = "strong" if abs(train_prod_corr) > 0.5 else "moderate" if abs(train_prod_corr) > 0.2 else "weak"
            insights.append(
                f"Employee AI training hours show a **{strength}** correlation "
                f"(r = {train_prod_corr:.2f}) with productivity gains, "
                f"{'supporting continued training investment.' if train_prod_corr > 0 else 'suggesting training alone may not drive productivity.'}"
            )

    # 5. Adoption trend over years
    by_year = df.groupby("year")["ai_adoption_level"].mean()
    if len(by_year) > 1:
        change = by_year.iloc[-1] - by_year.iloc[0]
        trend = "increased" if change > 0 else "decreased"
        insights.append(
            f"Average AI adoption {trend} by **{abs(change):.1%}** "
            f"from {by_year.index[0]} to {by_year.index[-1]} across the filtered data."
        )

    return insights
