"""
PAGE 4 — INDUSTRY DEEP DIVE
===========================================================================
Drop-in page for your Streamlit AI Adoption Dashboard.

HOW TO INTEGRATE
-----------------------------------------------------------------------
1. Save this file next to your main app (e.g. `page4_industry_deep_dive.py`).
2. In your main app file:

       from page4_industry_deep_dive import render_industry_deep_dive

3. Wherever your page router lives (the same session_state pattern you're
   already using for filter resets), call:

       render_industry_deep_dive(df, on_back=lambda: go_to_page("Page 3"))

   `on_back` is optional — pass your own navigation callback (whatever sets
   st.session_state.page in your router). If you don't pass one, a default
   generic PAGES list is used (see BACK BUTTON section at the bottom) —
   update the PAGES list there to match your actual page names.

ASSUMPTIONS (please confirm these match your data / spec)
-----------------------------------------------------------------------
* `df` is your main corporate_ai_adoption_dataset dataframe (the same one
  used on your other pages), with columns: industry, ai_investment_usd,
  automation_rate, cost_savings, revenue_impact, productivity_gain,
  ai_maturity_score.
* Your dataset has 10 industries, not just the 5 named in the spec.
  "Finance" in your spec maps to "Financial Services" in the data. All 10
  are included in the dropdown for completeness / robustness, with
  Financial Services selected by default.
* `revenue_impact` and `cost_savings` are raw USD in the dataset, not
  percentages. To express them as the "%" KPIs your spec asks for, they're
  computed here as a return-on-investment: (metric / avg AI investment) x
  100. Flag this if you intended something else (e.g. YoY revenue growth %
  from a different column).
* `automation_rate` and `productivity_gain` are stored as fractions
  (0–1) in the data, so they're multiplied by 100 for the "%" KPIs.
===========================================================================
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ---------------------------------------------------------------------
# Metric definitions (internal_key -> display label)
# ---------------------------------------------------------------------
INDUSTRY_METRICS = {
    "ai_investment_usd": "Total AI Investment",
    "revenue_growth_pct": "Revenue Growth (%)",
    "cost_savings_pct": "Cost Savings (%)",
    "productivity_gain_pct": "Productivity Increase (%)",
    "ai_maturity_score": "AI Maturity Score",
    "automation_rate_pct": "Automation Rate (%)",
}

ACCENT = "#6C5CE7"      # selected industry highlight
NEUTRAL = "#D3D3E0"     # all other industries
GOOD = "#00B894"
BAD = "#D63031"


# ---------------------------------------------------------------------
# Data prep (cached — recomputes only if df changes)
# ---------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def compute_industry_summary(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("industry").agg(
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


def _fmt_currency(v: float) -> str:
    if v >= 1e9:
        return f"${v/1e9:,.2f}B"
    if v >= 1e6:
        return f"${v/1e6:,.1f}M"
    return f"${v:,.0f}"


def _fmt_metric(key: str, val: float) -> str:
    if key == "ai_investment_usd":
        return _fmt_currency(val)
    if key == "ai_maturity_score":
        return f"{val:.2f}/10"
    return f"{val:.1f}%"


# ---------------------------------------------------------------------
# Chart builders
# ---------------------------------------------------------------------
def _build_comparison_bars(summary: pd.DataFrame, selected: str) -> go.Figure:
    fig = make_subplots(
        rows=2, cols=3,
        subplot_titles=list(INDUSTRY_METRICS.values()),
        horizontal_spacing=0.10, vertical_spacing=0.18,
    )
    positions = [(1, 1), (1, 2), (1, 3), (2, 1), (2, 2), (2, 3)]

    for (row_i, col_i), (key, label) in zip(positions, INDUSTRY_METRICS.items()):
        sorted_df = summary[["industry", key]].sort_values(key, ascending=True)
        colors = [ACCENT if ind == selected else NEUTRAL for ind in sorted_df["industry"]]
        fig.add_trace(
            go.Bar(
                x=sorted_df[key],
                y=sorted_df["industry"],
                orientation="h",
                marker_color=colors,
                showlegend=False,
                hovertemplate="%{y}: %{x:,.2f}<extra></extra>",
            ),
            row=row_i, col=col_i,
        )

    fig.update_layout(
        height=620,
        margin=dict(t=60, b=20, l=10, r=10),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(size=11),
    )
    return fig


def _build_radar(summary: pd.DataFrame, selected: str) -> go.Figure:
    keys = list(INDUSTRY_METRICS.keys())
    labels = list(INDUSTRY_METRICS.values())

    mins = summary[keys].min()
    maxs = summary[keys].max()
    norm = (summary[keys] - mins) / (maxs - mins).replace(0, 1) * 100

    sel_row = norm[summary["industry"] == selected].iloc[0]
    avg_row = norm.mean()

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=list(sel_row.values) + [sel_row.values[0]],
        theta=labels + [labels[0]],
        fill="toself",
        name=selected,
        line_color=ACCENT,
    ))
    fig.add_trace(go.Scatterpolar(
        r=list(avg_row.values) + [avg_row.values[0]],
        theta=labels + [labels[0]],
        fill="toself",
        name="All-Industry Average",
        line_color=NEUTRAL,
        opacity=0.6,
    ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
        showlegend=True,
        height=450,
        margin=dict(t=30, b=20, l=40, r=40),
        legend=dict(orientation="h", yanchor="bottom", y=-0.15),
    )
    return fig


def _build_heatmap(summary: pd.DataFrame) -> go.Figure:
    keys = list(INDUSTRY_METRICS.keys())
    labels = list(INDUSTRY_METRICS.values())

    ordered = summary.sort_values("ai_investment_usd", ascending=False)
    mins = ordered[keys].min()
    maxs = ordered[keys].max()
    norm = (ordered[keys] - mins) / (maxs - mins).replace(0, 1)

    text = ordered[keys].apply(
        lambda col: [_fmt_metric(col.name, v) for v in col]
    )

    fig = go.Figure(data=go.Heatmap(
        z=norm.values,
        x=labels,
        y=ordered["industry"],
        text=text.values,
        texttemplate="%{text}",
        textfont=dict(size=10),
        colorscale="Purples",
        colorbar=dict(title="Relative<br>Performance"),
        hovertemplate="%{y} — %{x}: %{text}<extra></extra>",
    ))
    fig.update_layout(
        height=450,
        margin=dict(t=30, b=20, l=10, r=10),
    )
    return fig


# ---------------------------------------------------------------------
# Insight generator
# ---------------------------------------------------------------------
def _generate_insights(summary: pd.DataFrame, selected: str) -> list[str]:
    row = summary[summary["industry"] == selected].iloc[0]
    insights = []

    leader = summary.loc[summary["ai_investment_usd"].idxmax()]
    insights.append(
        f"💰 **{leader['industry']}** receives the highest total AI investment "
        f"at {_fmt_currency(leader['ai_investment_usd'])}."
    )

    leader = summary.loc[summary["productivity_gain_pct"].idxmax()]
    insights.append(
        f"⚙️ **{leader['industry']}** achieves the greatest productivity increase "
        f"at {leader['productivity_gain_pct']:.1f}%."
    )

    leader = summary.loc[summary["ai_maturity_score"].idxmax()]
    insights.append(
        f"🧠 **{leader['industry']}** has the highest AI maturity score "
        f"({leader['ai_maturity_score']:.2f}/10)."
    )

    leader = summary.loc[summary["automation_rate_pct"].idxmax()]
    insights.append(
        f"🤖 **{leader['industry']}** has the highest automation rate "
        f"at {leader['automation_rate_pct']:.1f}%."
    )

    for key, label in INDUSTRY_METRICS.items():
        avg = summary[key].mean()
        diff_pct = (row[key] - avg) / avg * 100 if avg != 0 else 0
        direction = "above" if diff_pct >= 0 else "below"
        arrow = "🔼" if diff_pct >= 0 else "🔽"
        insights.append(
            f"{arrow} **{selected}**'s {label} is **{abs(diff_pct):.1f}% {direction}** "
            f"the all-industry average."
        )

    return insights


# ---------------------------------------------------------------------
# Main render function
# ---------------------------------------------------------------------
def render_industry_deep_dive(df: pd.DataFrame, on_back=None):
    # ---- Back button (top of page, callback-based to avoid rerun glitches) ----
    def _go_back():
        if on_back is not None:
            on_back()
        else:
            # Generic fallback — replace with your actual page list/order
            pages = st.session_state.get(
                "app_pages",
                ["Overview", "Trends", "Industry Comparison",
                 "Industry Deep Dive", "Predictive Insights"],
            )
            current = st.session_state.get("page", "Industry Deep Dive")
            if current in pages and pages.index(current) > 0:
                st.session_state.page = pages[pages.index(current) - 1]

    st.button("⬅ Back", key="page4_back_btn", on_click=_go_back)

    st.markdown("## 🏭 Industry Deep Dive")
    st.caption("Explore and compare AI adoption performance across industries.")

    summary = compute_industry_summary(df)
    industries = sorted(summary["industry"].unique().tolist())
    default_idx = industries.index("Financial Services") if "Financial Services" in industries else 0

    selected_industry = st.sidebar.selectbox(
        "🏭 Select Industry",
        options=industries,
        index=default_idx,
        key="page4_industry_filter",
    )

    row = summary[summary["industry"] == selected_industry].iloc[0]
    overall_avg = {key: summary[key].mean() for key in INDUSTRY_METRICS}

    # ---- KPI Cards -----------------------------------------------------
    st.markdown(f"### {selected_industry} — Key Metrics")
    kpi_cols = st.columns(6)
    kpi_meta = [
        ("ai_investment_usd", "💰 Total AI Investment"),
        ("revenue_growth_pct", "📈 Revenue Growth"),
        ("cost_savings_pct", "💵 Cost Savings"),
        ("productivity_gain_pct", "⚙️ Productivity Increase"),
        ("ai_maturity_score", "🧠 AI Maturity Score"),
        ("automation_rate_pct", "🤖 Automation Rate"),
    ]
    for col, (key, label) in zip(kpi_cols, kpi_meta):
        delta_pct = (row[key] - overall_avg[key]) / overall_avg[key] * 100 if overall_avg[key] != 0 else 0
        col.metric(label, _fmt_metric(key, row[key]), f"{delta_pct:+.1f}% vs avg")

    st.divider()

    # ---- Horizontal bar comparison --------------------------------------
    st.markdown("### 📊 Selected Industry vs. All Others")
    st.plotly_chart(_build_comparison_bars(summary, selected_industry), use_container_width=True)

    # ---- Radar + Heatmap side by side -----------------------------------
    st.markdown("### 🕸️ Multi-Metric Profile & Industry Heatmap")
    col_radar, col_heat = st.columns(2)
    with col_radar:
        st.plotly_chart(_build_radar(summary, selected_industry), use_container_width=True)
    with col_heat:
        st.plotly_chart(_build_heatmap(summary), use_container_width=True)

    st.divider()

    # ---- Business insights ----------------------------------------------
    st.markdown("### 💡 Business Questions Answered")
    for line in _generate_insights(summary, selected_industry):
        st.markdown(f"- {line}")
