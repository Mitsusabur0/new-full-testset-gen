from __future__ import annotations

import ast
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import streamlit as st
import altair as alt


APP_DIR = Path(__file__).parent
PARQUET_PATH = APP_DIR / "testset_results.parquet"
CSV_PATH = APP_DIR / "evaluation_set_full.csv"


st.set_page_config(
    page_title="RAG Evaluation Dashboard",
    page_icon=":bar_chart:",
    layout="wide",
    initial_sidebar_state="expanded",
)


# --- HELPERS ---

def _to_list(value: Any) -> list:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if hasattr(value, "tolist"):
        try:
            return value.tolist()
        except Exception:
            pass
    if isinstance(value, str):
        try:
            parsed = ast.literal_eval(value)
            if isinstance(parsed, list):
                return parsed
        except Exception:
            return [value]
    return [value]


def _coerce_numeric(df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


@st.cache_data(show_spinner=False)
def load_data() -> pd.DataFrame:
    if PARQUET_PATH.exists():
        df = pd.read_parquet(PARQUET_PATH)
    elif CSV_PATH.exists():
        try:
            df = pd.read_csv(CSV_PATH, encoding="utf-8")
        except UnicodeDecodeError:
            df = pd.read_csv(CSV_PATH, encoding="latin-1")
    else:
        return pd.DataFrame()

    list_cols = ["reference_contexts", "retrieved_contexts", "retrieved_file"]
    for col in list_cols:
        if col in df.columns:
            df[col] = df[col].apply(_to_list)

    metric_cols = [
        "custom_hit_rate",
        "custom_mrr",
        "custom_precision_at_k",
        "custom_recall_at_k",
        "deepeval_contextual_precision",
        "deepeval_contextual_recall",
        "deepeval_contextual_relevancy",
        "ragas_context_precision",
        "ragas_context_recall",
        "ragas_context_entity_recall",
    ]
    df = _coerce_numeric(df, metric_cols)

    # Convenience columns
    if "custom_hit_rate" in df.columns:
        df["is_hit"] = df["custom_hit_rate"].fillna(0).astype(int)
    if "custom_mrr" in df.columns:
        df["mrr_bucket"] = pd.cut(
            df["custom_mrr"].fillna(0),
            bins=[-0.01, 0, 0.33, 0.66, 1.0],
            labels=["0", "0–0.33", "0.33–0.66", "0.66–1.0"],
        )

    return df


# --- STYLING ---

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap');

html, body, [class*="css"]  {
    font-family: 'Space Grotesk', sans-serif;
}

/* Background & Main Containers */
.reportview-container {
    background: radial-gradient(1200px 600px at 20% -10%, #e7f3ff 0%, #f7f9fc 55%, #ffffff 100%);
}
section.main > div {
    padding-top: 1.5rem;
}

/* Hero Section */
.hero {
    padding: 1.5rem 2rem;
    border-radius: 16px;
    background: #0f172a;
    color: #ffffff;
    border: 1px solid rgba(255,255,255,0.1);
    box-shadow: 0 20px 60px rgba(15, 23, 42, 0.2);
    margin-bottom: 2rem;
}
.hero h1 {
    font-size: 2.2rem;
    margin-bottom: 0.5rem;
    font-weight: 700;
}
.hero p {
    margin-top: 0.25rem;
    color: rgba(255, 255, 255, 0.8);
    font-size: 1rem;
}

/* Typography & Badges */
.metric-pill {
    display: inline-block;
    padding: 0.25rem 0.75rem;
    border-radius: 999px;
    font-size: 0.8rem;
    background: #eff6ff;
    color: #1d4ed8;
    border: 1px solid #dbeafe;
    margin-right: 0.5rem;
    font-weight: 500;
}
.code-block {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.85rem;
    background: #1e293b;
    color: #e2e8f0;
    padding: 1rem;
    border-radius: 8px;
    line-height: 1.5;
}

/* Section Headers */
.section-header {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    font-size: 1.4rem;
    font-weight: 700;
    margin-bottom: 1rem;
    margin-top: 1rem;
    color: whitesmoke;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}
.global-metrics-container .section-header {
    color: whitesmoke;
}
.section-indicator {
    width: 12px;
    height: 12px;
    border-radius: 50%;
}

.section-header::after {
    content: "";
    flex: 1;
    height: 2px;
    margin-left: 0.8rem;
    # background: linear-gradient(10deg, currentColor, rgba(148, 163, 184, 0));
    # background: linear-gradient(90deg, currentColor, rgba(148, 163, 184, 0));
}
.global-metrics-title {
    color: whitesmoke;
}

/* Theme Colors */
.theme-custom { color: #2563eb; }
.bg-custom { background-color: #2563eb; }

.theme-deepeval { color: #0f766e; }
.bg-deepeval { background-color: #0f766e; }

.theme-ragas { color: #ea580c; }
.bg-ragas { background-color: #ea580c; }

/* Force Global Metrics headers to whitesmoke regardless of theme class */
.section-header.global-metrics-title {
    color: whitesmoke !important;
}

/* Adjustments for Streamlit Buttons to look like Cards in the Grid */
div[data-testid="stVerticalBlock"] button {
    text-align: left;
    height: auto !important;
    padding-top: 0.8rem !important;
    padding-bottom: 0.8rem !important;
    border-radius: 8px !important;
}

/* KPI Cards */
.kpi-card {
    padding: 0.9rem 1rem;
    border-radius: 12px;
    background: transparent;
    border: 1px solid rgba(148, 163, 184, 0.2);
    # box-shadow: 0 12px 24px rgba(15, 23, 42, 0.12);
    box-shadow: 2px 2px 8px black;
    margin-bottom: 0.7rem;
    max-width: 280px;
}
.kpi-row {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
    gap: 1.4rem;
    margin-bottom: 1.25rem;
}
.global-metrics-container {
    max-width: 1200px;
    margin: 0 auto;
}
.kpi-card-no-border {
    border: none;
}
.kpi-card-custom {
}
.kpi-card-deepeval {
}
.kpi-card-ragas {
}

.kpi-card-score {
    border-width: 2px;
}

.kpi-label {
    font-size: calc(0.78rem + 4px);
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: lightgrey;
}
.kpi-value {
    font-size: calc(1.6rem + 4px);
    font-weight: 600;
    color: #0f172a;
    color: whitesmoke;
}

.section-card {
    padding: 1.2rem;
    border-radius: 16px;
    background: #ffffff;
    border: 1px solid #e5e7eb;
    box-shadow: 0 14px 30px rgba(15, 23, 42, 0.05);
}
.section-title {
    font-size: 0.85rem;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    font-weight: 600;
    color: #475569;
    display: flex;
    align-items: center;
    gap: 0.6rem;
    margin-bottom: 0.4rem;
}
.section-title::before {
    content: "";
    width: 12px;
    height: 12px;
    border-radius: 999px;
    background: currentColor;
}
.section-divider {
    height: 1px;
    margin: 0.75rem 0 1.1rem 0;
    background: linear-gradient(90deg, currentColor, rgba(148, 163, 184, 0));
    border: none;
}
.section-custom {
    color: #1d4ed8;
}
.section-deepeval {
    color: #0f766e;
}
.section-ragas {
    color: #ea580c;
}

/* Global Metrics Toggle */
.global-metrics-toggle {
    margin: 0.25rem 0 1.25rem 0;
}
.global-metrics-toggle div[role="radiogroup"] {
    display: flex;
    gap: 0.6rem;
    flex-wrap: wrap;
}
.global-metrics-toggle label {
    border: 1px solid rgba(148, 163, 184, 0.35);
    border-radius: 999px;
    padding: 0.35rem 0.9rem;
    background: rgba(255, 255, 255, 0.7);
    color: #0f172a;
    cursor: pointer;
    font-weight: 600;
    letter-spacing: 0.02em;
}
.global-metrics-toggle label:has(input:checked) {
    background: #0f172a;
    color: #ffffff;
    border-color: #0f172a;
    box-shadow: 0 6px 16px rgba(15, 23, 42, 0.25);
}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# --- COMPONENTS ---

def render_hero(df: pd.DataFrame) -> None:
    total = len(df)
    styles = df["query_style"].nunique() if "query_style" in df.columns else 0
    files = df["source_file"].nunique() if "source_file" in df.columns else 0
    st.markdown(
        f"""
        <div class="hero">
            <h1>RAG Evaluation Dashboard</h1>
            <p>End-to-end view of retrieval quality, LLM grounding, and testcase-level diagnostics.</p>
            <div style="margin-top: 1rem;">
                <span class="metric-pill">{total} testcases</span>
                <span class="metric-pill">{styles} query styles</span>
                <span class="metric-pill">{files} source files</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_query_style_chart(
    df: pd.DataFrame,
    metric_col: str,
    metric_label: str,
    color_hex: str,
    y_domain: tuple[float, float] = (0, 1)
) -> None:
    """Renders a bar chart for a single metric across query styles."""
    if "query_style" not in df.columns or metric_col not in df.columns:
        st.info(f"Data for {metric_label} is not available.")
        return

    # Aggregate
    grouped = df.groupby("query_style")[metric_col].mean().reset_index()
    
    chart = (
        alt.Chart(grouped)
        .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4, size=40)
        .encode(
            x=alt.X("query_style:N", title=None, axis=alt.Axis(labelAngle=0, labelFontWeight="bold")),
            y=alt.Y(
                f"{metric_col}:Q",
                title=metric_label,
                scale=alt.Scale(domain=y_domain),
                axis=alt.Axis(format=".0%"),
            ),
            color=alt.value(color_hex),
            tooltip=[
                alt.Tooltip("query_style", title="Style"),
                alt.Tooltip(f"{metric_col}:Q", title="Score", format=".3f")
            ],
        )
        .properties(height=320, title=f"Average {metric_label} by Query Style")
        .configure_axis(grid=False)
        .configure_view(strokeWidth=0)
    )
    st.altair_chart(chart, use_container_width=True)


def render_interactive_metric_group(
    df: pd.DataFrame,
    group_id: str,
    title: str,
    metrics: list[tuple[str, str, str]],  # (Label, Column, Format)
    theme_color: str,
    theme_class: str
) -> None:
    """
    Renders a 2-column layout:
    Left: Vertical list of clickable metric buttons (KPIs).
    Right: Chart for the selected metric.
    """
    
    # 1. Header
    st.markdown(
        f"""
        <div class="section-header {theme_class}">
            <div class="section-indicator {theme_class.replace('theme-', 'bg-')}"></div>
            {title}
        </div>
        """, 
        unsafe_allow_html=True
    )

    # 2. Filter available metrics
    available_metrics = [m for m in metrics if m[1] in df.columns]
    
    if not available_metrics:
        st.warning(f"No metrics available for {title}.")
        return

    # 3. Session State Initialization for this group
    state_key = f"selected_metric_{group_id}"
    if state_key not in st.session_state:
        st.session_state[state_key] = available_metrics[0][1]

    # 4. Layout
    col_kpis, col_spacer, col_chart = st.columns([1, 0.1, 2.5])

    with col_kpis:
        st.markdown(f"<div style='margin-bottom: 0.5rem; font-size: 0.85rem; color: #64748b; text-transform: uppercase; letter-spacing: 0.05em;'>Select Metric</div>", unsafe_allow_html=True)
        for label, col_name, fmt in available_metrics:
            # Calculate Value
            val = df[col_name].mean()
            if pd.isna(val):
                val_str = "N/A"
            elif fmt == "percent":
                val_str = f"{val:.1%}"
            else:
                val_str = f"{val:.3f}"

            # Determine button state
            is_active = (st.session_state[state_key] == col_name)
            btn_type = "primary" if is_active else "secondary"
            
            # The button text acts as the "Card" content
            # We use newlines to separate Label and Value
            button_label = f"{label}\n{val_str}"
            
            if st.button(button_label, key=f"btn_{group_id}_{col_name}", type=btn_type, use_container_width=True):
                st.session_state[state_key] = col_name
                st.rerun()

    with col_chart:
        current_metric_col = st.session_state[state_key]
        # Find label for title
        current_label = next((m[0] for m in available_metrics if m[1] == current_metric_col), current_metric_col)
        
        render_query_style_chart(
            df=df,
            metric_col=current_metric_col,
            metric_label=current_label,
            color_hex=theme_color
        )
    
    st.markdown("---")


def _format_metric_value(value: float | None, fmt: str) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    if fmt == "percent":
        return f"{value:.1%}"
    return f"{value:.3f}"


def render_global_metrics_overview_tab(df: pd.DataFrame) -> None:
    def value_to_color(value: float | None) -> str | None:
        if value is None or pd.isna(value):
            return None
        v = max(0.0, min(1.0, float(value)))
        # Skew toward red so mid values stay warmer (more red) than green.
        v = v ** 2
        r = int(round(214 * (1 - v) + 16 * v))  # 0->#d65151, 1->#10b981
        g = int(round(81 * (1 - v) + 185 * v))
        b = int(round(81 * (1 - v) + 129 * v))
        return f"rgb({r}, {g}, {b})"

    def render_kpi_section(
        title: str,
        metrics: list[tuple[str, str, str]],
        tone: str,
    ) -> None:
        cards = []
        numeric_values = []
        for label, col_name, fmt in metrics:
            if col_name not in df.columns:
                continue
            value = df[col_name].mean()
            color = value_to_color(value)
            if not pd.isna(value):
                numeric_values.append(value)
            cards.append(
                {
                    "label": label,
                    "value": _format_metric_value(value, fmt),
                    "color": color,
                    "class": f"kpi-card kpi-card-{tone} kpi-card-no-border",
                }
            )

        if numeric_values:
            score_val = sum(numeric_values) / len(numeric_values)
            cards.append(
                {
                    "label": "OVERALL SCORE",
                    "value": _format_metric_value(score_val, "percent"),
                    "color": value_to_color(score_val),
                    "class": f"kpi-card kpi-card-{tone} kpi-card-score",
                }
            )

        if not cards:
            st.info(f"No metrics available for {title}.")
            return

        cards_html = "".join(
            f"<div class=\"{c['class']}\">"
            f"<div class=\"kpi-label\">{c['label']}</div>"
            f"<div class=\"kpi-value\""
            f"{' style=\"color: ' + c['color'] + ';\"' if c.get('color') else ''}"
            f">{c['value']}</div>"
            f"</div>"
            for c in cards
        )

        st.markdown(
            f"<div class=\"section-header theme-{tone} global-metrics-title\">{title}</div>",
            unsafe_allow_html=True,
        )
        st.markdown(f"<div class=\"kpi-row\">{cards_html}</div>", unsafe_allow_html=True)

    # Custom Metrics
    custom_metrics = [
        ("Hit Rate", "custom_hit_rate", "percent"),
        ("MRR", "custom_mrr", "float"),
        ("Precision@K", "custom_precision_at_k", "float"),
        ("Recall@K", "custom_recall_at_k", "float"),
    ]
    # DeepEval Metrics
    deepeval_metrics = [
        ("Contextual Precision", "deepeval_contextual_precision", "float"),
        ("Contextual Recall", "deepeval_contextual_recall", "float"),
        ("Contextual Relevancy", "deepeval_contextual_relevancy", "float"),
    ]

    # RAGAS Metrics
    ragas_metrics = [
        ("Context Precision", "ragas_context_precision", "float"),
        ("Context Recall", "ragas_context_recall", "float"),
        ("Context Entity Recall", "ragas_context_entity_recall", "float"),
    ]
    options = ["Custom", "DeepEval", "RAGAS"]

    st.markdown("<div class=\"global-metrics-container\">", unsafe_allow_html=True)
    st.markdown("<div class=\"global-metrics-toggle\">", unsafe_allow_html=True)
    st.radio(
        "Global Metrics Section",
        options,
        index=0,
        horizontal=True,
        label_visibility="collapsed",
        key="global_metrics_section",
    )
    selected = st.session_state["global_metrics_section"]
    st.markdown("</div>", unsafe_allow_html=True)

    if selected == "Custom":
        render_kpi_section("Custom Metrics", custom_metrics, "custom")
    elif selected == "DeepEval":
        render_kpi_section("DeepEval Metrics", deepeval_metrics, "deepeval")
    else:
        render_kpi_section("RAGAS Metrics", ragas_metrics, "ragas")

    st.markdown("</div>", unsafe_allow_html=True)


def render_by_query_style_tab(df: pd.DataFrame) -> None:
    # 1. Custom Metrics
    custom_metrics = [
        ("Hit Rate", "custom_hit_rate", "percent"),
        ("MRR", "custom_mrr", "float"),
        ("Precision@K", "custom_precision_at_k", "float"),
        ("Recall@K", "custom_recall_at_k", "float"),
    ]
    render_interactive_metric_group(
        df, 
        group_id="custom", 
        title="Custom Metrics", 
        metrics=custom_metrics, 
        theme_color="#2563eb", # Blue
        theme_class="theme-custom"
    )

    # 2. DeepEval Metrics
    deepeval_metrics = [
        ("Contextual Precision", "deepeval_contextual_precision", "float"),
        ("Contextual Recall", "deepeval_contextual_recall", "float"),
        ("Contextual Relevancy", "deepeval_contextual_relevancy", "float"),
    ]
    render_interactive_metric_group(
        df, 
        group_id="deepeval", 
        title="DeepEval Metrics", 
        metrics=deepeval_metrics, 
        theme_color="#0f766e", # Teal
        theme_class="theme-deepeval"
    )

    # 3. RAGAS Metrics
    ragas_metrics = [
        ("Context Precision", "ragas_context_precision", "float"),
        ("Context Recall", "ragas_context_recall", "float"),
        ("Entity Recall", "ragas_context_entity_recall", "float"),
    ]
    render_interactive_metric_group(
        df, 
        group_id="ragas", 
        title="RAGAS Metrics", 
        metrics=ragas_metrics, 
        theme_color="#ea580c", # Orange
        theme_class="theme-ragas"
    )


def render_case_explorer(df: pd.DataFrame) -> None:
    st.markdown("### Testcase Explorer")

    # Table View
    display_cols = [
        col for col in ["user_input", "query_style", "source_file", "custom_mrr", "custom_hit_rate"]
        if col in df.columns
    ]
    
    selection = st.dataframe(
        df[display_cols],
        use_container_width=True,
        hide_index=True,
        on_select="rerun",
        selection_mode="single-row",
        height=300,
    )

    selected_idx = None
    if selection.selection and selection.selection.rows:
        selected_row_idx = selection.selection.rows[0]
        selected_idx = df.index[selected_row_idx]

    if selected_idx is None:
        st.info("Select a row above to view details.")
        return

    row = df.loc[selected_idx]

    st.markdown("---")
    
    # Details Grid
    col1, col2 = st.columns([1.5, 1])

    with col1:
        st.subheader("Prompt & Outputs")
        st.markdown(f"**User Input:**\n\n{row.get('user_input', '—')}")
        st.markdown(f"**Expected:**\n\n{row.get('expected_output', '—')}")
        st.markdown(f"**Actual:**\n\n{row.get('actual_output', '—')}")

    with col2:
        st.subheader("Scores")
        # Helper to render a score row
        def score_row(label, val, fmt=".4f"):
            if pd.isna(val): return
            v_str = f"{val:{fmt}}" if isinstance(val, float) else str(val)
            st.markdown(f"**{label}:** `{v_str}`")

        score_row("MRR", row.get("custom_mrr"))
        score_row("Hit Rate", row.get("custom_hit_rate"))
        st.caption("DeepEval")
        score_row("Precision", row.get("deepeval_contextual_precision"))
        score_row("Recall", row.get("deepeval_contextual_recall"))
        st.caption("RAGAS")
        score_row("Precision", row.get("ragas_context_precision"))
        score_row("Recall", row.get("ragas_context_recall"))

    st.markdown("---")

    # Contexts
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Ground Truth Context")
        gt = row.get("reference_contexts", [])
        if gt:
            st.markdown(f"<div class='code-block'>{"\n\n".join(str(x) for x in gt)}</div>", unsafe_allow_html=True)
        else:
            st.text("No ground truth context.")

    with c2:
        st.subheader("Retrieved Contexts")
        ret = row.get("retrieved_contexts", [])
        ret_files = row.get("retrieved_file", [])
        
        if not ret:
            st.text("No contexts retrieved.")
        else:
            for i, txt in enumerate(ret):
                fname = ret_files[i] if i < len(ret_files) else "Unknown"
                st.markdown(f"**Rank {i+1}** (File: {fname})")
                st.markdown(f"<div class='code-block' style='font-size:0.8em; margin-bottom:0.5rem;'>{txt}</div>", unsafe_allow_html=True)


def render_filters(df: pd.DataFrame) -> pd.DataFrame:
    with st.sidebar:
        st.header("Filters")
        
        # Text Search
        search = st.text_input("Search User Input", placeholder="Keywords...")
        
        # Style Filter
        if "query_style" in df.columns:
            all_styles = sorted(df["query_style"].dropna().unique())
            sel_styles = st.multiselect("Query Style", all_styles, default=all_styles)
        else:
            sel_styles = []
            
        # File Filter
        if "source_file" in df.columns:
            all_files = sorted(df["source_file"].dropna().unique())
            sel_files = st.multiselect("Source File", all_files, default=all_files)
        else:
            sel_files = []

        # Metric Filter
        min_mrr = st.slider("Min MRR", 0.0, 1.0, 0.0)
        failures_only = st.toggle("Show Failures Only (Hit=0)", False)

    # Apply Logic
    out = df.copy()
    if search:
        out = out[out["user_input"].str.contains(search, case=False, na=False)]
    if sel_styles:
        out = out[out["query_style"].isin(sel_styles)]
    if sel_files:
        out = out[out["source_file"].isin(sel_files)]
    
    if "custom_mrr" in out.columns:
        out = out[out["custom_mrr"].fillna(0) >= min_mrr]
    
    if failures_only and "custom_hit_rate" in out.columns:
        out = out[out["custom_hit_rate"].fillna(0) == 0]
        
    return out


def main() -> None:
    df = load_data()
    if df.empty:
        st.error("No data found. Please add 'testset_results.parquet' or 'evaluation_set_full.csv' to the app directory.")
        return

    render_hero(df)
    
    filtered_df = render_filters(df)
    
    if filtered_df.empty:
        st.warning("No data matches the selected filters.")
        return

    tab1, tab2, tab3 = st.tabs(["Global Metrics", "By Query Style", "Testcase Explorer"])
    
    with tab1:
        render_global_metrics_overview_tab(filtered_df)
        
    with tab2:
        render_by_query_style_tab(filtered_df)

    with tab3:
        render_case_explorer(filtered_df)


if __name__ == "__main__":
    main()
