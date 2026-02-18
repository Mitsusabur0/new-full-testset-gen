from __future__ import annotations

import ast
import html
import json
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import streamlit as st
import altair as alt


APP_DIR = Path(__file__).parent
DATASETS_DIR = APP_DIR / "complete_datasets"
METRICS_PATH = APP_DIR / "metrics.json"


st.set_page_config(
    page_title="Panel de evaluación RAG",
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
def load_data(dataset_path: Path) -> pd.DataFrame:
    if not dataset_path.exists():
        return pd.DataFrame()
    df = pd.read_parquet(dataset_path)

    list_cols = ["reference_contexts", "retrieved_contexts", "retrieved_file"]
    for col in list_cols:
        if col in df.columns:
            df[col] = df[col].apply(_to_list)

    metric_cols = [
        "custom_hit_rate",
        "custom_mrr",
        "custom_precision_at_k",
        "custom_recall_at_k",
    ]
    df = _coerce_numeric(df, metric_cols)

    # Convenience columns
    if "custom_hit_rate" in df.columns:
        df["is_hit"] = df["custom_hit_rate"].fillna(0).astype(int)
    if "custom_mrr" in df.columns:
        df["mrr_bucket"] = pd.cut(
            df["custom_mrr"].fillna(0),
            bins=[-0.01, 0, 0.33, 0.66, 1.0],
            labels=["0", "0-0.33", "0.33-0.66", "0.66-1.0"],
        )

    return df


@st.cache_data(show_spinner=False)
def load_metric_descriptions() -> dict[str, dict[str, str]]:
    if not METRICS_PATH.exists():
        return {}
    try:
        payload = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}

    out: dict[str, dict[str, str]] = {}
    for group, items in payload.items():
        if not isinstance(items, list):
            continue
        group_map: dict[str, str] = {}
        for item in items:
            if not isinstance(item, dict):
                continue
            name = str(item.get("name", "")).strip()
            desc = str(item.get("description", "")).strip()
            if name and desc:
                group_map[name] = desc
        if group_map:
            out[group] = group_map
    return out


# --- STYLING ---

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap');

html, body, [class*="css"]  {
    font-family: 'Space Grotesk', sans-serif;
}

/* Background & Main Containers */
.reportview-container {
    background: var(--secondary-background-color);
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
.kpi-row-vertical {
    display: flex;
    flex-direction: column;
    gap: 0.9rem;
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

.kpi-card-score {
    border-width: 2px;
}
.kpi-card-highlight {
    border-color: #22c55e !important;
    box-shadow: 0 0 0 2px rgba(34, 197, 94, 0.35);
}

.kpi-label {
    font-size: 1rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: lightgrey;
}
.kpi-help {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    text-transform: none;
    width: 16px;
    height: 16px;
    margin-left: 6px;
    border-radius: 999px;
    border: 1px solid rgba(148, 163, 184, 0.7);
    color: #cbd5f5;
    font-size: 0.7rem;
    line-height: 1;
    cursor: help;
    position: relative;
}
.kpi-help::after {
    content: attr(data-tooltip);
    position: absolute;
    left: 50%;
    top: calc(100% + 8px);
    transform: translateX(-50%);
    background: rgba(15, 23, 42, 0.96);
    color: #e2e8f0;
    padding: 0.5rem 0.65rem;
    border-radius: 8px;
    font-size: 1rem;
    line-height: 1.25;
    letter-spacing: 0.01em;
    width: max-content;
    max-width: 280px;
    white-space: normal;
    opacity: 0;
    pointer-events: none;
    transition: opacity 140ms ease, transform 140ms ease;
    box-shadow: 0 10px 24px rgba(15, 23, 42, 0.35);
    z-index: 5;
}
.kpi-help::before {
    content: "";
    position: absolute;
    left: 50%;
    top: calc(100% + 2px);
    transform: translateX(-50%);
    border-width: 6px;
    border-style: solid;
    border-color: rgba(15, 23, 42, 0.96) transparent transparent transparent;
    opacity: 0;
    transition: opacity 140ms ease, transform 140ms ease;
    z-index: 6;
}
.kpi-help:hover::after,
.kpi-help:hover::before {
    opacity: 1;
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

/* Testcase Explorer */
.case-block {
    border: 1px solid rgba(148, 163, 184, 0.35);
    border-radius: 14px;
    background: #0f172a;
    box-shadow: 0 18px 36px rgba(15, 23, 42, 0.12);
    overflow: hidden;
}
.case-section {
    padding: 0.95rem 1rem;
}
.case-section + .case-section {
    border-top: 1px solid rgba(148, 163, 184, 0.25);
}
.case-title {
    font-size: 0.75rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #cbd5f5;
    font-weight: 600;
    margin-bottom: 0.35rem;
}
.case-body {
    white-space: pre-wrap;
    color: #e2e8f0;
}
.hit-rank {
    color: #16a34a;
    font-weight: 700;
}
.hit-file {
    color: #22c55e;
    font-weight: 600;
}
.hit-badge {
    display: inline-block;
    margin-left: 0.5rem;
    padding: 0.15rem 0.45rem;
    border-radius: 999px;
    background: rgba(34, 197, 94, 0.15);
    border: 1px solid rgba(34, 197, 94, 0.45);
    color: #22c55e;
    font-size: 0.7rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
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
            <h1>Panel de evaluación RAG</h1>
            <p>Vista integral de la calidad de recuperación.</p>
            <div style="margin-top: 1rem;">
                <span class="metric-pill">{total} casos de prueba</span>
                <span class="metric-pill">{styles} estilos de consulta</span>
                <span class="metric-pill">{files} archivos fuente</span>
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
        st.info(f"No hay datos disponibles para {metric_label}.")
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
                alt.Tooltip("query_style", title="Estilo"),
                alt.Tooltip(f"{metric_col}:Q", title="Puntuación", format=".3f")
            ],
        )
        .properties(height=320, title=f"Promedio de {metric_label} por estilo de consulta")
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
    theme_class: str,
    header_class: str | None = None,
    indicator_color: str | None = None,
) -> None:
    """
    Renders a 2-column layout:
    Left: Vertical list of clickable metric buttons (KPIs).
    Right: Chart for the selected metric.
    """
    
    # 1. Header
    header_classes = "section-header"
    if theme_class:
        header_classes = f"{header_classes} {theme_class}"
    if header_class:
        header_classes = f"{header_classes} {header_class}"

    indicator_class = theme_class.replace("theme-", "bg-") if theme_class else ""
    indicator_style = f" style=\"background: {indicator_color};\"" if indicator_color else ""
    indicator_html = ""
    if indicator_class or indicator_color:
        indicator_html = f"<div class=\"section-indicator {indicator_class}\"{indicator_style}></div>"

    st.markdown(
        f"""
        <div class="{header_classes}">
            {indicator_html}
            {title}
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 2. Filter available metrics
    available_metrics = [m for m in metrics if m[1] in df.columns]
    
    if not available_metrics:
        st.warning(f"No hay métricas disponibles para {title}.")
        return

    # 3. Session State Initialization for this group
    state_key = f"selected_metric_{group_id}"
    if state_key not in st.session_state:
        st.session_state[state_key] = available_metrics[0][1]

    # 4. Layout
    col_kpis, col_spacer, col_chart = st.columns([1, 0.1, 2.5])

    with col_kpis:
        st.markdown(f"<div style='margin-bottom: 0.5rem; font-size: 0.85rem; color: #64748b; text-transform: uppercase; letter-spacing: 0.05em;'>Seleccionar métrica</div>", unsafe_allow_html=True)
        for label, col_name, fmt in available_metrics:
            # Calculate Value
            val = df[col_name].mean()
            if pd.isna(val):
                val_str = "N/D"
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
        return "N/D"
    if fmt == "percent":
        return f"{value:.1%}"
    return f"{value:.3f}"


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


def _available_datasets() -> list[Path]:
    if not DATASETS_DIR.exists():
        return []
    return sorted(DATASETS_DIR.glob("*.parquet"))


def _render_kpi_cards(
    df: pd.DataFrame,
    title: str,
    metrics: list[tuple[str, str, str]],
    tone: str,
    descriptions: dict[str, str],
    score_first: bool = False,
    row_class: str = "kpi-row",
    highlight_score: bool = False,
) -> None:
    label_aliases: dict[str, dict[str, str]] = {
        "custom": {
            "Tasa de aciertos": "Hit Rate",
            "Precisión@K": "Precision@K",
            "Cobertura@K": "Recall@K",
            "MRR": "MRR",
        },
    }

    cards = []
    numeric_values = []
    for label, col_name, fmt in metrics:
        if col_name not in df.columns:
            continue
        value = df[col_name].mean()
        color = value_to_color(value)
        if not pd.isna(value):
            numeric_values.append(value)
        description = descriptions.get(label, "")
        if not description:
            alias = label_aliases.get(tone, {}).get(label, "")
            if alias:
                description = descriptions.get(alias, "")
        cards.append(
            {
                "label": label,
                "description": description,
                "value": _format_metric_value(value, fmt),
                "color": color,
                "class": f"kpi-card kpi-card-{tone} kpi-card-no-border",
            }
        )

    if numeric_values:
        score_val = sum(numeric_values) / len(numeric_values)
        score_card = {
            "label": "PUNTAJE GENERAL",
            "description": "Promedio de las métricas disponibles en esta sección.",
            "value": _format_metric_value(score_val, "percent"),
            "color": value_to_color(score_val),
            "class": f"kpi-card kpi-card-{tone} kpi-card-score",
        }
        if highlight_score:
            score_card["class"] = f"{score_card['class']} kpi-card-highlight"
        if score_first:
            cards = [score_card] + cards
        else:
            cards.append(score_card)

    if not cards:
        st.info(f"No hay métricas disponibles para {title}.")
        return

    def label_with_help(label: str, description: str) -> str:
        if not description:
            return label
        safe_desc = html.escape(description)
        return f"{label}<span class=\"kpi-help\" data-tooltip=\"{safe_desc}\">?</span>"

    cards_html = "".join(
        f"<div class=\"{c['class']}\">"
        f"<div class=\"kpi-label\">"
        f"{label_with_help(c['label'], c.get('description', ''))}"
        f"</div>"
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
    st.markdown(f"<div class=\"{row_class}\">{cards_html}</div>", unsafe_allow_html=True)


def render_global_metrics_overview_tab(df: pd.DataFrame) -> None:
    custom_metrics = [
        ("MRR", "custom_mrr", "float"),
        ("Tasa de aciertos", "custom_hit_rate", "percent"),
        ("Precisión@K", "custom_precision_at_k", "float"),
        ("Cobertura@K", "custom_recall_at_k", "float"),
    ]
    metric_descriptions = load_metric_descriptions()

    st.markdown("<div class=\"global-metrics-container\">", unsafe_allow_html=True)
    _render_kpi_cards(
        df,
        "Métricas tradicionales",
        custom_metrics,
        "custom",
        metric_descriptions.get("custom", {}),
    )
    st.markdown("---")
    render_interactive_metric_group(
        df,
        group_id="global_custom",
        title="por estilo de consulta",
        metrics=custom_metrics,
        theme_color="#2563eb",
        theme_class="",
        header_class="global-metrics-title",
        indicator_color="whitesmoke",
    )
    st.markdown("</div>", unsafe_allow_html=True)
def render_by_query_style_tab(df: pd.DataFrame) -> None:
    custom_metrics = [
        ("MRR", "custom_mrr", "float"),
        ("Tasa de aciertos", "custom_hit_rate", "percent"),
        ("Precisión@K", "custom_precision_at_k", "float"),
        ("Cobertura@K", "custom_recall_at_k", "float"),
    ]
    render_interactive_metric_group(
        df,
        group_id="custom",
        title="Métricas tradicionales",
        metrics=custom_metrics,
        theme_color="#2563eb",
        theme_class="theme-custom"
    )
def render_case_explorer(df: pd.DataFrame) -> None:
    st.markdown("### Explorador de casos de prueba")

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
        st.info("Selecciona una fila arriba para ver los detalles.")
        return

    row = df.loc[selected_idx]

    st.markdown("---")
    
    # Details Grid
    col1, col_gap, col2 = st.columns([1.5, 0.15, 1])

    with col1:
        st.subheader("Prompt y salidas")
        user_input = row.get("user_input", "—")
        expected_output = row.get("expected_output", "—")
        actual_output = row.get("actual_output", "—")

        case_html = (
            "<div class=\"case-block\">"
            f"<div class=\"case-section\"><div class=\"case-title\">Entrada del usuario</div>"
            f"<div class=\"case-body\">{html.escape(str(user_input))}</div></div>"
            f"<div class=\"case-section\"><div class=\"case-title\">Salida esperada</div>"
            f"<div class=\"case-body\">{html.escape(str(expected_output))}</div></div>"
            f"<div class=\"case-section\"><div class=\"case-title\">Salida real</div>"
            f"<div class=\"case-body\">{html.escape(str(actual_output))}</div></div>"
            "</div>"
        )
        st.markdown(case_html, unsafe_allow_html=True)

    with col2:
        st.subheader("Puntuaciones")
        # Helper to render a score row
        def score_row(label, val, fmt=".4f"):
            if pd.isna(val): return
            v_str = f"{val:{fmt}}" if isinstance(val, float) else str(val)
            try:
                color = value_to_color(val)
            except (TypeError, ValueError):
                color = None
            color_style = f" style=\"color: {color}; font-weight: 600;\"" if color else ""
            st.markdown(
                f"**{label}:** <span{color_style}>{html.escape(v_str)}</span>",
                unsafe_allow_html=True,
            )

        score_row("MRR", row.get("custom_mrr"))
        score_row("Tasa de aciertos", row.get("custom_hit_rate"))

    st.markdown("---")

    # Contexts
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Contexto de referencia")
        gt = row.get("reference_contexts", [])
        if gt:
            st.markdown(f"<div class='code-block'>{"\n\n".join(str(x) for x in gt)}</div>", unsafe_allow_html=True)
        else:
            st.text("No hay contexto de referencia.")

    with c2:
        st.subheader("Contextos recuperados")
        ret = row.get("retrieved_contexts", [])
        ret_files = row.get("retrieved_file", [])
        source_file = str(row.get("source_file", "")).strip()
        
        if not ret:
            st.text("No se recuperaron contextos.")
        else:
            for i, txt in enumerate(ret):
                fname = ret_files[i] if i < len(ret_files) else "Desconocido"
                is_hit = False
                if source_file and fname:
                    is_hit = source_file in str(fname)
                rank_class = "hit-rank" if is_hit else ""
                file_class = "hit-file" if is_hit else ""
                badge_html = "<span class=\"hit-badge\">Acierto</span>" if is_hit else ""
                st.markdown(
                    f"<div class=\"{rank_class}\">Rango {i+1}{badge_html}</div>"
                    f"<div class=\"{file_class}\" style=\"margin-bottom: 0.35rem;\">(Archivo: {html.escape(str(fname))})</div>",
                    unsafe_allow_html=True,
                )
                st.markdown(
                    f"<div class='code-block' style='font-size:0.8em; margin-bottom:0.5rem;'>{html.escape(str(txt))}</div>",
                    unsafe_allow_html=True,
                )


def render_compare_datasets_tab() -> None:
    dataset_paths = _available_datasets()
    if not dataset_paths:
        st.error(f"No se encontraron archivos parquet en {DATASETS_DIR}")
        return

    options = [p.name for p in dataset_paths]
    metric_descriptions = load_metric_descriptions()

    custom_metrics = [
        ("MRR", "custom_mrr", "float"),
        ("Tasa de aciertos", "custom_hit_rate", "percent"),
        ("Precisión@K", "custom_precision_at_k", "float"),
        ("Cobertura@K", "custom_recall_at_k", "float"),
    ]

    def compute_group_score(df: pd.DataFrame, metrics: list[tuple[str, str, str]]) -> float | None:
        values = []
        for _, col_name, _ in metrics:
            if col_name not in df.columns:
                continue
            val = df[col_name].mean()
            if not pd.isna(val):
                values.append(float(val))
        if not values:
            return None
        return sum(values) / len(values)

    def render_column(df: pd.DataFrame, highlight: dict[str, bool]) -> None:
        if df.empty:
            st.warning("No se encontraron datos en el conjunto seleccionado.")
            return

        st.markdown("<div style='height: 0.5rem;'></div>", unsafe_allow_html=True)

        _render_kpi_cards(
            df,
            "Métricas tradicionales",
            custom_metrics,
            "custom",
            metric_descriptions.get("custom", {}),
            score_first=True,
            row_class="kpi-row kpi-row-vertical",
            highlight_score=highlight.get("custom", False),
        )

    col_left, col_gap, col_right = st.columns([0.85, 0.3, 0.85])
    with col_left:
        left_name = st.selectbox(
            "Seleccionar conjunto de datos",
            options,
            key="compare_dataset_left",
        )
    with col_right:
        right_name = st.selectbox(
            "Seleccionar conjunto de datos",
            options,
            key="compare_dataset_right",
        )

    left_df = load_data(DATASETS_DIR / left_name)
    right_df = load_data(DATASETS_DIR / right_name)

    left_scores = {
        "custom": compute_group_score(left_df, custom_metrics),
    }
    right_scores = {
        "custom": compute_group_score(right_df, custom_metrics),
    }

    def _rounded_score(value: float | None) -> float | None:
        if value is None:
            return None
        return round(value * 100, 1)

    def highlight_map(
        scores_a: dict[str, float | None],
        scores_b: dict[str, float | None],
        side: str,
    ) -> dict[str, bool]:
        out: dict[str, bool] = {}
        for key in ("custom",):
            a = _rounded_score(scores_a.get(key))
            b = _rounded_score(scores_b.get(key))
            if a is None and b is None:
                out[key] = False
            elif b is None:
                out[key] = side == "left"
            elif a is None:
                out[key] = side == "right"
            elif a == b:
                out[key] = True
            else:
                out[key] = (side == "left" and a > b) or (side == "right" and b > a)
        return out

    left_highlight = highlight_map(left_scores, right_scores, "left")
    right_highlight = highlight_map(left_scores, right_scores, "right")

    with col_left:
        render_column(left_df, left_highlight)
    with col_right:
        render_column(right_df, right_highlight)
def select_dataset() -> Path | None:
    with st.sidebar:
        st.header("Conjunto de datos")
        if not DATASETS_DIR.exists():
            st.error(f"No se encontró la carpeta de conjuntos de datos: {DATASETS_DIR}")
            return None
        parquet_files = sorted(DATASETS_DIR.glob("*.parquet"))
        if not parquet_files:
            st.error(f"No se encontraron archivos parquet en {DATASETS_DIR}")
            return None
        options = [p.name for p in parquet_files]
        selected_name = st.selectbox("Seleccionar conjunto de datos", options)
        return DATASETS_DIR / selected_name


def render_filters(df: pd.DataFrame) -> pd.DataFrame:
    with st.sidebar:
        st.header("Filtros")

        # Style Filter
        if "query_style" in df.columns:
            all_styles = sorted(df["query_style"].dropna().unique())
            sel_styles = st.multiselect("Estilo de consulta", all_styles, default=all_styles)
        else:
            sel_styles = []
        failures_only = st.toggle("Mostrar solo fallos (Acierto=0)", False)

    # Apply Logic
    out = df.copy()
    if sel_styles:
        out = out[out["query_style"].isin(sel_styles)]

    if failures_only and "custom_hit_rate" in out.columns:
        out = out[out["custom_hit_rate"].fillna(0) == 0]
        
    return out


def main() -> None:
    dataset_path = select_dataset()
    if dataset_path is None:
        return

    df = load_data(dataset_path)
    if df.empty:
        st.error(f"No se encontraron datos en el conjunto de datos seleccionado: {dataset_path.name}")
        return

    render_hero(df)
    
    filtered_df = render_filters(df)
    
    if filtered_df.empty:
        st.warning("Ningún dato coincide con los filtros seleccionados.")
        return

    tab1, tab2, tab3 = st.tabs(["Métricas globales", "Explorador de casos de prueba", "Comparar datasets"])
    
    with tab1:
        render_global_metrics_overview_tab(filtered_df)

    with tab2:
        render_case_explorer(filtered_df)

    with tab3:
        render_compare_datasets_tab()


if __name__ == "__main__":
    main()




