"""
app_4.py — SINGLE-FILE RUNNER FOR PAGE 4 (Industry Deep Dive)
===========================================================================
Run with:

    py -m streamlit run app_4.py

Keep this file in the SAME FOLDER as:

    corporate_ai_adoption_dataset.csv

If the CSV isn't found next to this script, a file uploader will appear
so you can upload it manually instead.
===========================================================================
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path

st.set_page_config(
    page_title="AI Adoption Dashboard — Industry Deep Dive",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ===========================================================================
# COLOR PALETTE
# ===========================================================================
ACCENT = "#6C5CE7"       # selected industry highlight
ACCENT_DARK = "#4834D4"
ACCENT_LIGHT = "#A29BFE"
NEUTRAL = "#DCDCEA"      # all other industries
GOOD = "#00B894"
BAD = "#D63031"
BG_CARD = "#FFFFFF"
BG_SOFT = "#F6F5FC"
TEXT_MAIN = "#2D2B4E"
TEXT_MUTED = "#7A7898"

# ===========================================================================
# GLOBAL CSS — polished, card-based UI
# ===========================================================================
st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {{
        font-family: 'Inter', sans-serif;
    }}

    .main {{
        background: linear-gradient(180deg, {BG_SOFT} 0%, #FFFFFF 320px);
    }}

    /* ---- Hero header ---- */
    .hero {{
        background: linear-gradient(135deg, {ACCENT_DARK} 0%, {ACCENT} 55%, {ACCENT_LIGHT} 100%);
        border-radius: 20px;
        padding: 2rem 2.2rem;
        margin-bottom: 1.6rem;
        box-shadow: 0 10px 30px rgba(108, 92, 231, 0.25);
        color: white;
    }}
    .hero h1 {{
        margin: 0;
        font-size: 2rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        color: white;
    }}
    .hero p {{
        margin: 0.35rem 0 0 0;
        font-size: 1rem;
        color: rgba(255,255,255,0.88);
        font-weight: 400;
    }}
    .hero .badge {{
        display: inline-block;
        background: rgba(255,255,255,0.18);
        border: 1px solid rgba(255,255,255,0.35);
        padding: 0.25rem 0.8rem;
        border-radius: 999px;
        font-size: 0.8rem;
        font-weight: 600;
        margin-top: 0.9rem;
        color: white;
    }}

    /* ---- Section headers ---- */
    .section-title {{
        font-size: 1.25rem;
        font-weight: 700;
        color: {TEXT_MAIN};
        margin: 1.6rem 0 0.6rem 0;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }}
    .section-sub {{
        color: {TEXT_MUTED};
        font-size: 0.92rem;
        margin-bottom: 0.9rem;
    }}

    /* ---- KPI Cards ---- */
    .kpi-card {{
        background: {BG_CARD};
        border-radius: 16px;
        padding: 1.1rem 1.2rem;
        border: 1px solid #ECEBF7;
        box-shadow: 0 4px 14px rgba(108, 92, 231, 0.06);
        height: 100%;
        transition: transform 0.15s ease;
    }}
    .kpi-card:hover {{
        transform: translateY(-2px);
        box-shadow: 0 8px 22px rgba(108, 92, 231, 0.14);
    }}
    .kpi-label {{
        font-size: 0.8rem;
        font-weight: 600;
        color: {TEXT_MUTED};
        text-transform: uppercase;
        letter-spacing: 0.03em;
        margin-bottom: 0.35rem;
    }}
    .kpi-value {{
        font-size: 1.55rem;
        font-weight: 800;
        color: {TEXT_MAIN};
        line-height: 1.15;
    }}
    .kpi-delta-pos {{
        color: {GOOD};
        font-weight: 700;
        font-size: 0.85rem;
        margin-top: 0.3rem;
    }}
    .kpi-delta-neg {{
        color: {BAD};
        font-weight: 700;
        font-size: 0.85rem;
        margin-top: 0.3rem;
    }}

    /* ---- Insight cards ---- */
    .insight-card {{
        background: {BG_CARD};
        border-left: 4px solid {ACCENT};
        border-radius: 10px;
        padding: 0.85rem 1.1rem;
        margin-bottom: 0.6rem;
        box-shadow: 0 2px 8px rgba(108, 92, 231, 0.06);
        font-size: 0.95rem;
        color: {TEXT_MAIN};
    }}

    /* ---- Chart containers ---- */
    .chart-card {{
        background: {BG_CARD};
        border-radius: 16px;
        padding: 1rem 1.1rem 0.4rem 1.1rem;
        border: 1px solid #ECEBF7;
        box-shadow: 0 4px 14px rgba(108, 92, 231, 0.06);
        margin-bottom: 1.2rem;
    }}

    /* ---- Sidebar ---- */
    section[data-testid="stSidebar"] {{
        background: linear-gradient(180deg, #2D2B4E 0%, #4834D4 100%);
    }}
    section[data-testid="stSidebar"] * {{
        color: #EDEBFA !important;
    }}
    section[data-testid="stSidebar"] .stSelectbox label {{
        font-weight: 600;
    }}

    /* ---- Back button ---- */
    div[data-testid="stButton"] > button {{
        border-radius: 10px;
        border: 1px solid #ECEBF7;
        font-weight: 600;
        color: {TEXT_MAIN};
        background: white;
    }}
    div[data-testid="stButton"] > button:hover {{
        border-color: {ACCENT};
        color: {ACCENT_DARK};
    }}

    hr {{
        border-color: #ECEBF7 !important;
    }}
</style>
""", unsafe_allow_html=True)


# ===========================================================================
# METRIC DEFINITIONS
# ===========================================================================
INDUSTRY_METRICS = {
    "ai_investment_usd": "Total AI Investment",
    "revenue_growth_pct": "Revenue Growth (%)",
    "cost_savings_pct": "Cost Savings (%)",
    "productivity_gain_pct": "Productivity Increase (%)",
    "ai_maturity_score": "AI Maturity Score",
    "automation_rate_pct": "Automation Rate (%)",
}

KPI_ICONS = {
    "ai_investment_usd": "💰",
    "revenue_growth_pct": "📈",
    "cost_savings_pct": "💵",
    "productivity_gain_pct": "⚙️",
    "ai_maturity_score": "🧠",
    "automation_rate_pct": "🤖",
}


# ===========================================================================
# DATA LOADING
# ===========================================================================
@st.cache_data(show_spinner="Loading dataset...")
def load_data(path_or_buffer) -> pd.DataFrame:
    return pd.read_csv(path_or_buffer)


DEFAULT_CSV_NAME = "corporate_ai_adoption_dataset.csv"
csv_path = Path(__file__).parent / DEFAULT_CSV_NAME

if csv_path.exists():
    df = load_data(str(csv_path))
else:
    st.warning(
        f"Couldn't find `{DEFAULT_CSV_NAME}` next to this script. "
        f"Upload it below to continue."
    )
    uploaded = st.file_uploader(f"Upload {DEFAULT_CSV_NAME}", type="csv")
    if uploaded is None:
        st.stop()
    df = load_data(uploaded)


# ===========================================================================
# DATA PREP (cached — recomputes only if df changes)
# ===========================================================================
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


# ===========================================================================
# CHART BUILDERS
# ===========================================================================
def _build_comparison_bars(summary: pd.DataFrame, selected: str) -> go.Figure:
    fig = make_subplots(
        rows=2, cols=3,
        subplot_titles=list(INDUSTRY_METRICS.values()),
        horizontal_spacing=0.10, vertical_spacing=0.20,
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
                marker=dict(color=colors, line=dict(width=0)),
                showlegend=False,
                hovertemplate="%{y}: %{x:,.2f}<extra></extra>",
            ),
            row=row_i, col=col_i,
        )

    fig.update_layout(
        height=640,
        margin=dict(t=60, b=20, l=10, r=10),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(size=11, family="Inter, sans-serif", color=TEXT_MAIN),
    )
    fig.update_annotations(font=dict(size=12, family="Inter, sans-serif", color=TEXT_MAIN))
    fig.update_xaxes(showgrid=True, gridcolor="#EFEEF9", zeroline=False)
    fig.update_yaxes(showgrid=False)
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
        line=dict(color=ACCENT, width=2),
        fillcolor="rgba(108, 92, 231, 0.28)",
    ))
    fig.add_trace(go.Scatterpolar(
        r=list(avg_row.values) + [avg_row.values[0]],
        theta=labels + [labels[0]],
        fill="toself",
        name="All-Industry Average",
        line=dict(color="#B2B0C9", width=2, dash="dot"),
        fillcolor="rgba(178, 176, 201, 0.18)",
    ))
    fig.update_layout(
        polar=dict(
            bgcolor="rgba(0,0,0,0)",
            radialaxis=dict(visible=True, range=[0, 100], gridcolor="#EFEEF9"),
            angularaxis=dict(gridcolor="#EFEEF9"),
        ),
        showlegend=True,
        height=440,
        margin=dict(t=30, b=20, l=40, r=40),
        legend=dict(orientation="h", yanchor="bottom", y=-0.18),
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color=TEXT_MAIN, size=11),
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
        textfont=dict(size=10, family="Inter, sans-serif"),
        colorscale="Purples",
        colorbar=dict(title="Relative<br>Performance"),
        hovertemplate="%{y} — %{x}: %{text}<extra></extra>",
        xgap=3,
        ygap=3,
    ))
    fig.update_layout(
        height=440,
        margin=dict(t=30, b=20, l=10, r=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color=TEXT_MAIN, size=11),
    )
    return fig


# ===========================================================================
# INSIGHT GENERATOR
# ===========================================================================
def _generate_insights(summary: pd.DataFrame, selected: str) -> list[str]:
    row = summary[summary["industry"] == selected].iloc[0]
    insights = []

    leader = summary.loc[summary["ai_investment_usd"].idxmax()]
    insights.append(
        f"💰 <b>{leader['industry']}</b> receives the highest total AI investment "
        f"at {_fmt_currency(leader['ai_investment_usd'])}."
    )

    leader = summary.loc[summary["productivity_gain_pct"].idxmax()]
    insights.append(
        f"⚙️ <b>{leader['industry']}</b> achieves the greatest productivity increase "
        f"at {leader['productivity_gain_pct']:.1f}%."
    )

    leader = summary.loc[summary["ai_maturity_score"].idxmax()]
    insights.append(
        f"🧠 <b>{leader['industry']}</b> has the highest AI maturity score "
        f"({leader['ai_maturity_score']:.2f}/10)."
    )

    leader = summary.loc[summary["automation_rate_pct"].idxmax()]
    insights.append(
        f"🤖 <b>{leader['industry']}</b> has the highest automation rate "
        f"at {leader['automation_rate_pct']:.1f}%."
    )

    for key, label in INDUSTRY_METRICS.items():
        avg = summary[key].mean()
        diff_pct = (row[key] - avg) / avg * 100 if avg != 0 else 0
        direction = "above" if diff_pct >= 0 else "below"
        arrow = "🔼" if diff_pct >= 0 else "🔽"
        insights.append(
            f"{arrow} <b>{selected}</b>'s {label} is <b>{abs(diff_pct):.1f}% {direction}</b> "
            f"the all-industry average."
        )

    return insights


# ===========================================================================
# UI HELPERS
# ===========================================================================
def _section_title(icon: str, title: str, subtitle: str = ""):
    st.markdown(f'<div class="section-title">{icon} {title}</div>', unsafe_allow_html=True)
    if subtitle:
        st.markdown(f'<div class="section-sub">{subtitle}</div>', unsafe_allow_html=True)


def _kpi_card(col, icon: str, label: str, value: str, delta_pct: float):
    delta_class = "kpi-delta-pos" if delta_pct >= 0 else "kpi-delta-neg"
    arrow = "▲" if delta_pct >= 0 else "▼"
    with col:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">{icon} {label}</div>
            <div class="kpi-value">{value}</div>
            <div class="{delta_class}">{arrow} {abs(delta_pct):.1f}% vs avg</div>
        </div>
        """, unsafe_allow_html=True)


# ===========================================================================
# MAIN RENDER FUNCTION (Page 4 — Industry Deep Dive)
# ===========================================================================
def render_industry_deep_dive(df: pd.DataFrame, on_back=None):
    # ---- Back button ----
    def _go_back():
        if on_back is not None:
            on_back()
        else:
            pages = st.session_state.get(
                "app_pages",
                ["Overview", "Trends", "Industry Comparison",
                 "Industry Deep Dive", "Predictive Insights"],
            )
            current = st.session_state.get("page", "Industry Deep Dive")
            if current in pages and pages.index(current) > 0:
                st.session_state.page = pages[pages.index(current) - 1]

    st.button("⬅  Back", key="page4_back_btn", on_click=_go_back)

    summary = compute_industry_summary(df)
    industries = sorted(summary["industry"].unique().tolist())
    default_idx = industries.index("Financial Services") if "Financial Services" in industries else 0

    # ---- Sidebar ----
    st.sidebar.markdown("### 🏭 Industry Deep Dive")
    st.sidebar.markdown("Filter the dashboard by industry to compare performance side-by-side.")
    selected_industry = st.sidebar.selectbox(
        "Select Industry",
        options=industries,
        index=default_idx,
        key="page4_industry_filter",
    )
    st.sidebar.markdown("---")
    st.sidebar.markdown(
        f"<span style='font-size:0.82rem; opacity:0.8;'>"
        f"Dataset: <b>corporate_ai_adoption_dataset.csv</b><br>"
        f"Industries tracked: <b>{len(industries)}</b></span>",
        unsafe_allow_html=True,
    )

    # ---- Hero header ----
    st.markdown(f"""
    <div class="hero">
        <h1>🏭 Industry Deep Dive</h1>
        <p>Explore and compare AI adoption performance across industries.</p>
        <span class="badge">📌 Currently viewing: {selected_industry}</span>
    </div>
    """, unsafe_allow_html=True)

    row = summary[summary["industry"] == selected_industry].iloc[0]
    overall_avg = {key: summary[key].mean() for key in INDUSTRY_METRICS}

    # ---- KPI Cards ----
    _section_title("📌", f"{selected_industry} — Key Metrics")
    kpi_cols = st.columns(6)
    for col, (key, label) in zip(kpi_cols, INDUSTRY_METRICS.items()):
        delta_pct = (row[key] - overall_avg[key]) / overall_avg[key] * 100 if overall_avg[key] != 0 else 0
        _kpi_card(col, KPI_ICONS[key], label, _fmt_metric(key, row[key]), delta_pct)

    st.write("")

    # ---- Horizontal bar comparison ----
    _section_title("📊", "Selected Industry vs. All Others",
                    "Highlighted bars show where the selected industry ranks on every metric.")
    st.markdown('<div class="chart-card">', unsafe_allow_html=True)
    st.plotly_chart(_build_comparison_bars(summary, selected_industry), use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # ---- Radar + Heatmap side by side ----
    _section_title("🕸️", "Multi-Metric Profile & Industry Heatmap")
    col_radar, col_heat = st.columns(2)
    with col_radar:
        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        st.plotly_chart(_build_radar(summary, selected_industry), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with col_heat:
        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        st.plotly_chart(_build_heatmap(summary), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # ---- Business insights ----
    _section_title("💡", "Business Questions Answered")
    for line in _generate_insights(summary, selected_industry):
        st.markdown(f'<div class="insight-card">{line}</div>', unsafe_allow_html=True)


# ===========================================================================
# STANDALONE RUN (no other pages to go back to)
# ===========================================================================
def _no_back_available():
    st.session_state["_back_clicked"] = True


if st.session_state.get("_back_clicked"):
    st.info("This is a standalone preview of Page 4 — there's no other page to go back to yet.")
    st.session_state["_back_clicked"] = False

render_industry_deep_dive(df, on_back=_no_back_available)
