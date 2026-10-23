"""
AirWise India - Distributed Air Quality Analytics and AQI Prediction Using Apache Spark

Streamlit dashboard, step 2: real data connection plus the Overview, AQI Explorer,
City Comparison and Pollutant Analysis pages. Only your real data is shown.

Data: data/processed/aqi_clean.parquet
Outputs: outputs/analytics (tables) and outputs/charts (saved charts)

Run from the project folder:
    streamlit run app.py
"""

from __future__ import annotations

import calendar
import html
import inspect
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
PROJECT_DIR = Path(__file__).resolve().parent
PARQUET_PATH = PROJECT_DIR / "data" / "processed" / "aqi_clean.parquet"
ANALYTICS_DIR = PROJECT_DIR / "outputs" / "analytics"
CHARTS_DIR = PROJECT_DIR / "outputs" / "charts"

PAGES = [
    "Overview",
    "AQI Explorer",
    "City Comparison",
    "Pollutant Analysis",
    "AQI Category Prediction",
    "Numerical AQI Estimation",
    "Model Performance",
    "About Project",
]

PAGE_SUBTITLES = {
    "Overview": "Distributed air-quality analytics and AQI prediction for Indian cities, built on Apache Spark MLlib.",
    "AQI Explorer": "Filter the dataset and explore AQI and pollutant trends over time.",
    "City Comparison": "Compare air quality between cities side by side.",
    "Pollutant Analysis": "Understand which pollutants drive AQI and how they relate to each other.",
    "AQI Category Prediction": "Predict the AQI category from pollutant readings.",
    "Numerical AQI Estimation": "Estimate the numerical AQI value from pollutant readings.",
    "Model Performance": "Evaluate the trained Spark MLlib models.",
    "About Project": "Problem, objectives, architecture and technology stack.",
}

PAGE_PLANS = {
    "AQI Category Prediction": "Enter pollutant values and get a category prediction from your saved classifier.",
    "Numerical AQI Estimation": "Enter pollutant values and get an AQI estimate from your saved regression model.",
    "Model Performance": "Accuracy, F1, confusion matrix, R2, RMSE and MAE for the trained models.",
    "About Project": "Problem statement, architecture, dataset source, tools and methodology.",
}

POLLUTANTS = ["PM2.5", "PM10", "NO", "NO2", "NOx", "NH3", "CO", "SO2", "O3", "Benzene", "Toluene", "Xylene"]

BUCKET_ORDER = ["Good", "Satisfactory", "Moderate", "Poor", "Very Poor", "Severe"]
BUCKET_COLORS = {
    "Good": "#22c55e",
    "Satisfactory": "#a3e635",
    "Moderate": "#facc15",
    "Poor": "#fb923c",
    "Very Poor": "#ef4444",
    "Severe": "#9f1239",
}

FREQ = {"Daily": "D", "Weekly": "W", "Monthly": "MS", "Yearly": "YS"}
MIN_READINGS_PER_CITY = 30
POOR_THRESHOLD = 200  # AQI above this is "Poor" or worse on the Indian CPCB scale

TABLE_EXT = {".csv", ".parquet"}
IMAGE_EXT = {".png", ".jpg", ".jpeg", ".webp", ".gif"}
HTML_EXT = {".html", ".htm"}

SKIP_DIRS = {
    "venv", "env", "node_modules", "__pycache__", "site-packages",
    "spark-warehouse", "metastore_db", "checkpoint", "checkpoints",
}

TEAL = "#2dd4bf"
CYAN = "#22d3ee"
ORANGE = "#fb923c"
TEXT = "#e6f1ff"
MUTED = "#8ba3c0"
GRID = "rgba(139,163,192,0.12)"
FONT = "Inter, Segoe UI, system-ui, sans-serif"
PALETTE = ["#2dd4bf", "#38bdf8", "#a78bfa", "#f472b6", "#fbbf24", "#fb923c", "#34d399", "#60a5fa", "#e879f9", "#f87171"]

# --------------------------------------------------------------------------- #
# Theme (CSS)
# --------------------------------------------------------------------------- #
CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
:root {
  --bg: #0a1628; --sidebar: #07101f; --panel: #0f2240; --panel-2: #12294b;
  --border: rgba(45,212,191,.16); --teal: #2dd4bf; --cyan: #22d3ee;
  --text: #e6f1ff; --muted: #8ba3c0;
}
html, body, .stApp, [data-testid="stAppViewContainer"] {
  font-family: 'Inter', 'Segoe UI', system-ui, sans-serif; color: var(--text);
}
.stApp {
  background:
    radial-gradient(1200px 600px at 85% -10%, rgba(34,211,238,.10), transparent 60%),
    radial-gradient(900px 500px at -10% 110%, rgba(45,212,191,.08), transparent 55%),
    var(--bg);
}
[data-testid="stHeader"] { background: transparent; }
#MainMenu, footer { visibility: hidden; }
.block-container { padding-top: 2.2rem; padding-bottom: 3rem; max-width: 1280px; }

.stMarkdown, .stMarkdown p, [data-testid="stWidgetLabel"] p { color: var(--text); }
[data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] p { color: var(--muted); }

/* Sidebar */
[data-testid="stSidebar"] { background: var(--sidebar); border-right: 1px solid var(--border); }
[data-testid="stSidebar"] p, [data-testid="stSidebar"] label { color: var(--text); }
.brand { display: flex; gap: .75rem; align-items: center; padding: .25rem .25rem 1.2rem; }
.brand-logo {
  width: 40px; height: 40px; border-radius: 12px; display: flex; align-items: center; justify-content: center;
  background: linear-gradient(135deg, #2dd4bf, #22d3ee 60%, #38bdf8); color: #04121f; font-weight: 800;
  box-shadow: 0 6px 18px rgba(34,211,238,.25);
}
.brand-name { font-weight: 700; font-size: 1.08rem; letter-spacing: .2px; color: var(--text); }
.brand-tag { font-size: .72rem; color: var(--muted); }
.nav-label {
  font-size: .68rem; letter-spacing: .14em; text-transform: uppercase; color: var(--muted);
  margin: .6rem 0 .45rem .25rem;
}
[data-testid="stSidebar"] div[role="radiogroup"] { gap: .2rem; }
[data-testid="stSidebar"] div[role="radiogroup"] > label {
  padding: .55rem .75rem; border-radius: 10px; border: 1px solid transparent; width: 100%;
  cursor: pointer; transition: background .15s, border-color .15s;
}
[data-testid="stSidebar"] div[role="radiogroup"] > label:hover { background: rgba(45,212,191,.07); }
[data-testid="stSidebar"] div[role="radiogroup"] > label > div:first-child { display: none; }
[data-testid="stSidebar"] div[role="radiogroup"] > label:has(input:checked) {
  background: linear-gradient(90deg, rgba(45,212,191,.18), rgba(34,211,238,.05));
  border-color: rgba(45,212,191,.35);
}
[data-testid="stSidebar"] div[role="radiogroup"] > label:has(input:checked) p { color: var(--teal); font-weight: 600; }
.side-note { font-size: .78rem; color: var(--muted); line-height: 1.45; padding: 0 .25rem; word-break: break-word; }
.side-foot { font-size: .7rem; color: var(--muted); padding: 1.4rem .25rem 0; opacity: .8; }

/* Inputs */
div[data-baseweb="select"] > div, div[data-baseweb="input"] > div { background: var(--panel); border-color: var(--border); }
div[data-baseweb="select"] div { color: var(--text); }
[data-baseweb="popover"] ul { background: var(--panel); }
[data-baseweb="popover"] li, [data-baseweb="popover"] li * { color: var(--text); }
[data-baseweb="popover"] li:hover { background: rgba(45,212,191,.12); }
[data-baseweb="tag"] { background: rgba(45,212,191,.18); }
[data-baseweb="tag"] span { color: var(--text); }

/* Tabs */
button[data-baseweb="tab"] p { color: var(--muted); font-weight: 600; }
button[data-baseweb="tab"][aria-selected="true"] p { color: var(--teal); }
[data-baseweb="tab-highlight"] { background-color: var(--teal); }

/* Page header */
.eyebrow { font-size: .72rem; letter-spacing: .16em; text-transform: uppercase; color: var(--teal); font-weight: 600; }
.page-title { font-size: 2rem; font-weight: 700; margin: .15rem 0 .3rem; line-height: 1.2; color: var(--text); }
.page-sub { color: var(--muted); font-size: .98rem; margin-bottom: 1.5rem; max-width: 780px; }
.section-title { font-size: 1.05rem; font-weight: 600; color: var(--text); margin: .2rem 0 .6rem; }
.spacer { height: 1rem; }

/* KPI cards */
.kpi {
  background: linear-gradient(160deg, var(--panel-2), var(--panel)); border: 1px solid var(--border);
  border-radius: 16px; padding: 1.05rem 1.15rem 1rem 1.3rem; min-height: 116px; position: relative; overflow: hidden;
}
.kpi::before { content: ""; position: absolute; left: 0; top: 0; bottom: 0; width: 3px; background: linear-gradient(var(--teal), var(--cyan)); }
.kpi-label { font-size: .7rem; letter-spacing: .12em; text-transform: uppercase; color: var(--muted); font-weight: 600; }
.kpi-value { font-size: 1.75rem; font-weight: 700; color: var(--text); margin: .35rem 0 .15rem; line-height: 1.15; }
.kpi-value.small { font-size: 1.3rem; padding: .3rem 0 .2rem; }
.kpi-sub { font-size: .78rem; color: var(--muted); }

/* Chart cards */
[data-testid="stPlotlyChart"] {
  background: linear-gradient(160deg, var(--panel-2), var(--panel)); border: 1px solid var(--border);
  border-radius: 16px; padding: .5rem .5rem .25rem;
}

/* Generic panels */
.panel { background: linear-gradient(160deg, var(--panel-2), var(--panel)); border: 1px solid var(--border); border-radius: 16px; padding: 1.4rem 1.5rem; }
.panel h4 { margin: 0 0 .4rem; color: var(--text); font-size: 1.05rem; }
.panel p { margin: 0; color: var(--muted); line-height: 1.55; }
.pill {
  display: inline-block; margin-top: .9rem; padding: .25rem .7rem; border-radius: 999px; font-size: .72rem; font-weight: 600;
  color: var(--teal); background: rgba(45,212,191,.12); border: 1px solid rgba(45,212,191,.3);
}
"""


def inject_css() -> None:
    css = re.sub(r"\s*\n\s*", " ", CSS)
    st.markdown("<style>" + css + "</style>", unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# Data loading (your real files only)
# --------------------------------------------------------------------------- #
def _norm(name: object) -> str:
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


_CORE_LOOKUP = {"city": "City", "date": "Date", "datetime": "Date", "aqi": "AQI", "aqibucket": "AQI_Bucket"}
_POLLUTANT_LOOKUP = {_norm(p): p for p in POLLUTANTS}


def _canonical_columns(columns) -> dict:
    """Map raw column names (e.g. 'pm2_5', 'aqi_bucket') to the dashboard's standard names."""
    mapping: dict = {}
    for col in columns:
        key = _norm(col)
        target = _CORE_LOOKUP.get(key) or _POLLUTANT_LOOKUP.get(key)
        if target and target not in mapping.values():
            mapping[col] = target
    return mapping


def _read_csv(path, **kwargs) -> pd.DataFrame:
    try:
        return pd.read_csv(path, encoding="utf-8-sig", **kwargs)
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="latin-1", **kwargs)


def _read_one(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return _read_csv(path)


def _is_usable_csv(path: Path) -> bool:
    try:
        header = _read_csv(path, nrows=0)
    except Exception:
        return False
    return {"City", "AQI"} <= set(_canonical_columns(header.columns).values())


@st.cache_data(show_spinner=False)
def discover_csv_fallback(root: str) -> list[str]:
    """Only used when the parquet file is missing: look for a CSV with City and AQI columns."""
    root_path = Path(root)
    found: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(root_path):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS and not d.startswith(".") and d != "outputs"]
        if len(Path(dirpath).relative_to(root_path).parts) > 4:
            dirnames[:] = []
            continue
        for name in filenames:
            if name.lower().endswith(".csv") and not name.startswith("part-"):
                found.append(Path(dirpath) / name)
    usable = [p for p in found if _is_usable_csv(p)]
    usable.sort(key=lambda p: -p.stat().st_size)
    return [str(p) for p in usable]


def resolve_sources() -> list[str]:
    if PARQUET_PATH.exists():
        return [str(PARQUET_PATH)]
    return discover_csv_fallback(str(PROJECT_DIR))


@st.cache_data(show_spinner="Loading air-quality data...")
def load_data(path: str, modified: float) -> tuple[pd.DataFrame, list[str]]:
    raw = _read_one(Path(path)) if not Path(path).is_dir() else pd.read_parquet(path)
    if not isinstance(raw.index, pd.RangeIndex):
        raw = raw.reset_index()
    raw_columns = [str(c) for c in raw.columns]
    df = raw.rename(columns=_canonical_columns(raw.columns))
    keep = [c for c in ["City", "Date", "AQI", "AQI_Bucket", *POLLUTANTS] if c in df.columns]
    df = df[keep].copy()
    if "City" not in df.columns or "AQI" not in df.columns:
        return df, raw_columns

    df["City"] = df["City"].astype(str).str.strip()
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    for col in ["AQI", *POLLUTANTS]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "AQI_Bucket" in df.columns:
        df["AQI_Bucket"] = df["AQI_Bucket"].astype("string").str.strip()
    else:
        edges = [-float("inf"), 50, 100, 200, 300, 400, float("inf")]
        df["AQI_Bucket"] = pd.cut(df["AQI"], bins=edges, labels=BUCKET_ORDER).astype("string")
    return df, raw_columns


def aqi_label(value: float) -> str:
    for upper, label in ((50, "Good"), (100, "Satisfactory"), (200, "Moderate"), (300, "Poor"), (400, "Very Poor")):
        if value <= upper:
            return label
    return "Severe"


def rel_path(path: str) -> str:
    try:
        return str(Path(path).relative_to(PROJECT_DIR))
    except ValueError:
        return path


# --------------------------------------------------------------------------- #
# Pipeline outputs (outputs/analytics and outputs/charts)
# --------------------------------------------------------------------------- #
def list_tables(folder: Path) -> list[Path]:
    """CSV/Parquet files in a folder; Spark part-files are grouped under their parent folder."""
    if not folder.is_dir():
        return []
    items: dict[Path, None] = {}
    for p in sorted(folder.rglob("*")):
        if not p.is_file() or p.suffix.lower() not in TABLE_EXT or p.name.startswith(("_", ".")):
            continue
        items[p.parent if p.name.startswith("part-") else p] = None
    return list(items)


def list_charts(folder: Path) -> list[Path]:
    if not folder.is_dir():
        return []
    return sorted(
        p for p in folder.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXT | HTML_EXT and not p.name.startswith((".", "_"))
    )


@st.cache_data(ttl=300, show_spinner=False)
def read_table(path: str) -> pd.DataFrame:
    p = Path(path)
    if p.is_dir():
        parts = [q for q in sorted(p.glob("part-*")) if q.suffix.lower() in TABLE_EXT]
        return pd.concat([_read_one(q) for q in parts], ignore_index=True)
    return _read_one(p)


def show_html_file(path: Path, height: int = 560) -> None:
    """Embed one of your own saved HTML charts (newer Streamlit uses st.iframe, older uses components.html)."""
    if hasattr(st, "iframe"):
        st.iframe(path, height=height)
    else:
        import streamlit.components.v1 as components

        components.html(path.read_text(encoding="utf-8", errors="replace"), height=height, scrolling=True)


def show_table(df: pd.DataFrame) -> None:
    try:
        st.dataframe(df, width="stretch", hide_index=True)
    except Exception:
        st.dataframe(df, use_container_width=True, hide_index=True)


# --------------------------------------------------------------------------- #
# UI helpers
# --------------------------------------------------------------------------- #
_PLOTLY_WIDTH = (
    {"width": "stretch"}
    if "width" in inspect.signature(st.plotly_chart).parameters
    else {"use_container_width": True}
)


def show_chart(fig: go.Figure) -> None:
    st.plotly_chart(fig, config={"displayModeBar": False}, **_PLOTLY_WIDTH)


def style_fig(fig: go.Figure, title: str, height: int = 380, legend: bool = False) -> go.Figure:
    fig.update_layout(
        title=dict(text=title, x=0.02, xanchor="left", font=dict(size=15, color=TEXT)),
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=FONT, color="#cbd5e1", size=12),
        margin=dict(l=16, r=28, t=56, b=16),
        height=height,
        showlegend=legend,
        legend=dict(title_text="", font=dict(color="#cbd5e1")),
        hoverlabel=dict(bgcolor="#0f2342", bordercolor=CYAN, font=dict(color=TEXT)),
    )
    fig.update_xaxes(gridcolor=GRID, zeroline=False, linecolor=GRID)
    fig.update_yaxes(gridcolor=GRID, zeroline=False, linecolor=GRID)
    return fig


def page_header(page: str) -> None:
    st.markdown(
        '<div class="eyebrow">AirWise India</div>'
        f'<div class="page-title">{html.escape(page)}</div>'
        f'<div class="page-sub">{html.escape(PAGE_SUBTITLES[page])}</div>',
        unsafe_allow_html=True,
    )


def spacer() -> None:
    st.markdown('<div class="spacer"></div>', unsafe_allow_html=True)


def section_title(text: str) -> None:
    st.markdown(f'<div class="section-title">{html.escape(text)}</div>', unsafe_allow_html=True)


def kpi_card(label: str, value: str, sub: str, small: bool = False) -> str:
    size = " small" if small else ""
    return (
        '<div class="kpi">'
        f'<div class="kpi-label">{html.escape(label)}</div>'
        f'<div class="kpi-value{size}">{html.escape(value)}</div>'
        f'<div class="kpi-sub">{html.escape(sub)}</div>'
        "</div>"
    )


def kpi_row(cards: list[str]) -> None:
    for col, card in zip(st.columns(len(cards), gap="medium"), cards):
        col.markdown(card, unsafe_allow_html=True)


def render_empty_state() -> None:
    st.markdown(
        '<div class="panel"><h4>Dataset not found</h4>'
        f"<p>Expected the cleaned dataset at <b>{html.escape(rel_path(str(PARQUET_PATH)))}</b> "
        f"inside <b>{html.escape(str(PROJECT_DIR))}</b>. Run <b>streamlit run app.py</b> from the project folder, "
        "or check that the file exists.</p>"
        '<span class="pill">No data is generated or simulated</span></div>',
        unsafe_allow_html=True,
    )


def get_data(data_path: str | None) -> pd.DataFrame | None:
    """Load the dataset, or show a clear message and return None."""
    if not data_path:
        render_empty_state()
        return None
    try:
        df, raw_columns = load_data(data_path, Path(data_path).stat().st_mtime)
    except ImportError:
        st.error("Reading Parquet needs the pyarrow package. Run: pip install pyarrow")
        return None
    except Exception as exc:  # unreadable file
        st.error(f"Could not read {rel_path(data_path)}: {exc}")
        return None
    if "City" not in df.columns or "AQI" not in df.columns:
        st.error(
            "The dataset needs a City column and an AQI column. "
            f"Columns found in the file: {', '.join(raw_columns)}"
        )
        return None
    if df["AQI"].notna().sum() == 0:
        st.warning("The dataset has no numeric AQI values, so these pages cannot be drawn.")
        return None
    return df


# --------------------------------------------------------------------------- #
# Shared filters and statistics
# --------------------------------------------------------------------------- #
def available_pollutants(df: pd.DataFrame) -> list[str]:
    return [p for p in POLLUTANTS if p in df.columns and df[p].notna().any()]


def apply_date_filter(df: pd.DataFrame, key: str) -> pd.DataFrame:
    if "Date" not in df.columns or df["Date"].notna().sum() == 0:
        return df
    lo, hi = df["Date"].min().date(), df["Date"].max().date()
    picked = st.date_input("Date range", value=(lo, hi), min_value=lo, max_value=hi, key=key)
    if isinstance(picked, (tuple, list)) and len(picked) == 2:
        start, end = picked
        return df[(df["Date"] >= pd.Timestamp(start)) & (df["Date"] < pd.Timestamp(end) + pd.Timedelta(days=1))]
    return df


def city_stats(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Mean and reading count per city, limited to cities with enough readings."""
    stats = df.dropna(subset=[metric]).groupby("City")[metric].agg(["mean", "count"])
    ranked = stats[stats["count"] >= MIN_READINGS_PER_CITY]
    return ranked if not ranked.empty else stats


# --------------------------------------------------------------------------- #
# Charts
# --------------------------------------------------------------------------- #
def chart_trend(df: pd.DataFrame) -> go.Figure | None:
    if "Date" not in df.columns:
        return None
    series = df.dropna(subset=["Date", "AQI"]).set_index("Date")["AQI"].sort_index().resample("MS").mean().dropna()
    if series.empty:
        return None
    fig = go.Figure(
        go.Scatter(
            x=series.index, y=series.values, mode="lines",
            line=dict(color=TEAL, width=2.5, shape="spline", smoothing=0.6),
            fill="tozeroy", fillcolor="rgba(45,212,191,0.12)",
            hovertemplate="%{x|%b %Y}<br>Average AQI: %{y:.0f}<extra></extra>",
        )
    )
    fig.update_yaxes(title_text="Average AQI")
    return style_fig(fig, "Monthly average AQI")


def chart_categories(df: pd.DataFrame, title: str = "AQI category distribution") -> go.Figure | None:
    counts = df["AQI_Bucket"].dropna().value_counts()
    if counts.empty:
        return None
    order = [b for b in BUCKET_ORDER if b in counts.index] + [b for b in counts.index if b not in BUCKET_ORDER]
    fig = go.Figure(
        go.Pie(
            labels=order, values=[int(counts[b]) for b in order], hole=0.62, sort=False, textinfo="none",
            marker=dict(colors=[BUCKET_COLORS.get(b, CYAN) for b in order], line=dict(color="#0f2240", width=2)),
            hovertemplate="%{label}<br>%{value:,} records (%{percent})<extra></extra>",
        )
    )
    return style_fig(fig, title, legend=True)


def chart_city_bars(stats: pd.DataFrame, n: int, highest: bool, metric: str = "AQI") -> go.Figure:
    if highest:
        data = stats.nlargest(n, "mean").sort_values("mean")
        color = ORANGE
        title = f"Top {n} most polluted cities (mean AQI)" if metric == "AQI" else f"Top {n} cities by mean {metric}"
    else:
        data = stats.nsmallest(n, "mean").sort_values("mean", ascending=False)
        color = TEAL
        title = f"Top {n} cleanest cities (mean AQI)" if metric == "AQI" else f"Lowest {n} cities by mean {metric}"
    fig = go.Figure(
        go.Bar(
            x=data["mean"], y=data.index, orientation="h", marker=dict(color=color), customdata=data["count"],
            texttemplate="%{x:.0f}", textposition="outside", cliponaxis=False,
            hovertemplate=f"%{{y}}<br>Mean {metric}: %{{x:.1f}}<br>Readings: %{{customdata:,}}<extra></extra>",
        )
    )
    fig.update_yaxes(showgrid=False)
    return style_fig(fig, title, height=400)


def chart_timeseries(f: pd.DataFrame, metric: str, freq_label: str, by_city: bool) -> go.Figure | None:
    if "Date" not in f.columns:
        return None
    d = f.dropna(subset=["Date", metric])
    if d.empty:
        return None
    freq = FREQ[freq_label]
    if by_city:
        out = d.groupby(["City", pd.Grouper(key="Date", freq=freq)])[metric].mean().reset_index()
        fig = px.line(out, x="Date", y=metric, color="City", color_discrete_sequence=PALETTE)
        fig.update_traces(line=dict(width=2))
    else:
        out = d.groupby(pd.Grouper(key="Date", freq=freq))[metric].mean().reset_index()
        fig = px.line(out, x="Date", y=metric)
        fig.update_traces(line=dict(color=TEAL, width=2.5), fill="tozeroy", fillcolor="rgba(45,212,191,0.10)")
    fig.update_yaxes(title_text=f"Average {metric}")
    fig.update_xaxes(title_text="")
    return style_fig(fig, f"{freq_label} average {metric}", height=400, legend=by_city)


def chart_month_pattern(f: pd.DataFrame, metric: str) -> go.Figure | None:
    if "Date" not in f.columns:
        return None
    d = f.dropna(subset=["Date", metric])
    if d.empty:
        return None
    monthly = d.groupby(d["Date"].dt.month)[metric].mean().reindex(range(1, 13))
    fig = go.Figure(
        go.Bar(
            x=list(calendar.month_abbr)[1:], y=monthly.values, marker=dict(color=TEAL),
            hovertemplate=f"%{{x}}<br>Average {metric}: %{{y:.1f}}<extra></extra>",
        )
    )
    fig.update_yaxes(title_text=f"Average {metric}")
    return style_fig(fig, f"Seasonal pattern: average {metric} by month")


def chart_histogram(f: pd.DataFrame) -> go.Figure | None:
    d = f.dropna(subset=["AQI"])
    if d.empty:
        return None
    present = [b for b in BUCKET_ORDER if b in set(d["AQI_Bucket"].dropna())]
    fig = px.histogram(
        d, x="AQI", color="AQI_Bucket", nbins=50,
        color_discrete_map=BUCKET_COLORS, category_orders={"AQI_Bucket": present},
    )
    fig.update_yaxes(title_text="Records")
    return style_fig(fig, "AQI distribution", legend=True)


def chart_mean_by_city(f: pd.DataFrame, metric: str) -> go.Figure | None:
    stats = f.dropna(subset=[metric]).groupby("City")[metric].agg(["mean", "count"])
    if stats.empty:
        return None
    stats = stats.sort_values("mean", ascending=False)
    colors = [BUCKET_COLORS[aqi_label(v)] for v in stats["mean"]] if metric == "AQI" else TEAL
    fig = go.Figure(
        go.Bar(
            x=stats.index, y=stats["mean"], marker=dict(color=colors), customdata=stats["count"],
            texttemplate="%{y:.0f}", textposition="outside", cliponaxis=False,
            hovertemplate=f"%{{x}}<br>Mean {metric}: %{{y:.1f}}<br>Readings: %{{customdata:,}}<extra></extra>",
        )
    )
    fig.update_yaxes(title_text=f"Mean {metric}")
    return style_fig(fig, f"Mean {metric} by city")


def chart_box_by_city(f: pd.DataFrame, metric: str) -> go.Figure | None:
    d = f.dropna(subset=[metric])
    if d.empty:
        return None
    fig = px.box(d, x="City", y=metric, color="City", color_discrete_sequence=PALETTE, points=False)
    fig.update_xaxes(title_text="")
    return style_fig(fig, f"{metric} distribution by city")


def chart_category_share(f: pd.DataFrame) -> go.Figure | None:
    d = f.dropna(subset=["AQI_Bucket"])
    if d.empty:
        return None
    share = pd.crosstab(d["City"], d["AQI_Bucket"], normalize="index") * 100
    order = [b for b in BUCKET_ORDER if b in share.columns] + [b for b in share.columns if b not in BUCKET_ORDER]
    fig = go.Figure()
    for bucket in order:
        fig.add_trace(
            go.Bar(
                y=share.index, x=share[bucket], name=bucket, orientation="h",
                marker=dict(color=BUCKET_COLORS.get(bucket, CYAN)),
                hovertemplate=f"%{{y}}<br>{bucket}: %{{x:.1f}}%<extra></extra>",
            )
        )
    fig.update_layout(barmode="stack")
    fig.update_xaxes(title_text="Share of records (%)", range=[0, 100])
    return style_fig(fig, "AQI category share by city", legend=True)


def chart_corr_with_aqi(corr: pd.DataFrame) -> go.Figure | None:
    if "AQI" not in corr.columns or len(corr) < 2:
        return None
    series = corr["AQI"].drop("AQI").dropna().sort_values()
    if series.empty:
        return None
    fig = go.Figure(
        go.Bar(
            x=series.values, y=series.index, orientation="h",
            marker=dict(color=[TEAL if v >= 0 else ORANGE for v in series.values]),
            texttemplate="%{x:.2f}", textposition="outside", cliponaxis=False,
            hovertemplate="%{y}<br>Correlation with AQI: %{x:.2f}<extra></extra>",
        )
    )
    fig.update_xaxes(range=[min(-0.1, series.min() - 0.15), 1.1], title_text="Pearson correlation")
    fig.update_yaxes(showgrid=False)
    return style_fig(fig, "Correlation of each pollutant with AQI", height=400)


def chart_heatmap(corr: pd.DataFrame) -> go.Figure:
    fig = go.Figure(
        go.Heatmap(
            z=corr.values, x=corr.columns, y=corr.index, zmin=-1, zmax=1,
            colorscale=[[0, ORANGE], [0.5, "#0f2240"], [1, TEAL]],
            texttemplate="%{z:.2f}", textfont=dict(size=10),
            hovertemplate="%{y} vs %{x}<br>Correlation: %{z:.2f}<extra></extra>",
            colorbar=dict(thickness=10, outlinewidth=0),
        )
    )
    fig.update_yaxes(autorange="reversed", showgrid=False)
    fig.update_xaxes(showgrid=False)
    return style_fig(fig, "Correlation heatmap", height=460)


def chart_scatter(f: pd.DataFrame, pollutant: str) -> tuple[go.Figure | None, str]:
    d = f.dropna(subset=[pollutant, "AQI"])
    if len(d) < 3:
        return None, ""
    total = len(d)
    sample = d.sample(n=4000, random_state=42) if total > 4000 else d
    present = [b for b in BUCKET_ORDER if b in set(sample["AQI_Bucket"].dropna())]
    fig = px.scatter(
        sample, x=pollutant, y="AQI", color="AQI_Bucket", opacity=0.55,
        color_discrete_map=BUCKET_COLORS, category_orders={"AQI_Bucket": present},
    )
    fig.update_traces(marker=dict(size=5))
    if d[pollutant].nunique() > 1:
        slope, intercept = np.polyfit(d[pollutant], d["AQI"], 1)
        xs = np.array([d[pollutant].min(), d[pollutant].max()])
        fig.add_trace(go.Scatter(x=xs, y=slope * xs + intercept, mode="lines", name="Linear fit",
                                 line=dict(color=CYAN, width=2, dash="dash")))
    note = f"Showing a random sample of {len(sample):,} of {total:,} records." if total > 4000 else ""
    return style_fig(fig, f"{pollutant} vs AQI", legend=True), note


def chart_completeness(f: pd.DataFrame, columns: list[str]) -> go.Figure:
    pct = (f[columns].notna().mean() * 100).sort_values()
    fig = go.Figure(
        go.Bar(
            x=pct.values, y=pct.index, orientation="h", marker=dict(color=CYAN),
            texttemplate="%{x:.0f}%", textposition="outside", cliponaxis=False,
            hovertemplate="%{y}<br>%{x:.1f}% of records have a value<extra></extra>",
        )
    )
    fig.update_xaxes(range=[0, 112], title_text="Records with a value (%)")
    fig.update_yaxes(showgrid=False)
    return style_fig(fig, "Data completeness", height=400)


def maybe(fig: go.Figure | None, message: str = "Not enough data for this chart with the current filters.") -> None:
    if fig is None:
        st.info(message)
    else:
        show_chart(fig)


# --------------------------------------------------------------------------- #
# Pages
# --------------------------------------------------------------------------- #
def render_pipeline_outputs() -> None:
    section_title("Pipeline outputs")
    tab_tables, tab_charts = st.tabs(["Analytics tables", "Saved charts"])

    with tab_tables:
        tables = list_tables(ANALYTICS_DIR)
        if not tables:
            st.info(f"No tables found in {rel_path(str(ANALYTICS_DIR))}.")
        else:
            choice = st.selectbox("Table", tables, format_func=lambda p: rel_path(str(p)), key="ov_table")
            try:
                table = read_table(str(choice))
                st.caption(f"{len(table):,} rows, {len(table.columns)} columns")
                show_table(table)
            except Exception as exc:
                st.error(f"Could not read {rel_path(str(choice))}: {exc}")

    with tab_charts:
        charts = list_charts(CHARTS_DIR)
        if not charts:
            st.info(f"No charts found in {rel_path(str(CHARTS_DIR))}.")
        else:
            choice = st.selectbox("Chart", charts, format_func=lambda p: rel_path(str(p)), key="ov_chart")
            try:
                if choice.suffix.lower() in HTML_EXT:
                    show_html_file(choice)
                else:
                    st.image(str(choice))
            except Exception as exc:
                st.error(f"Could not display {rel_path(str(choice))}: {exc}")


def render_overview(data_path: str | None) -> None:
    page_header("Overview")
    df = get_data(data_path)
    if df is None:
        return

    ranked = city_stats(df, "AQI")
    mean_aqi = float(df["AQI"].mean())
    period, period_sub = "n/a", "no date column"
    if "Date" in df.columns and df["Date"].notna().any():
        start, end = df["Date"].min(), df["Date"].max()
        period = f"{start.year} - {end.year}" if start.year != end.year else str(start.year)
        period_sub = f"{start:%b %Y} to {end:%b %Y}"
    worst = ranked["mean"].idxmax()
    kpi_row([
        kpi_card("Records", f"{len(df):,}", "rows in dataset"),
        kpi_card("Cities", f"{df['City'].nunique():,}", "monitoring locations"),
        kpi_card("Period", period, period_sub),
        kpi_card("Mean AQI", f"{mean_aqi:.0f}", aqi_label(mean_aqi)),
        kpi_card("Most polluted city", str(worst), f"mean AQI {ranked.loc[worst, 'mean']:.0f}", small=True),
    ])
    spacer()

    left, right = st.columns([2, 1], gap="medium")
    with left:
        trend = chart_trend(df)
        if trend is not None:
            show_chart(trend)
            st.caption("Average across all reporting cities; the number of reporting cities can change from month to month.")
        else:
            st.info("A Date column is needed to draw the AQI trend.")
    with right:
        maybe(chart_categories(df))

    spacer()
    n = min(10, max(1, len(ranked) // 2))
    left, right = st.columns(2, gap="medium")
    with left:
        show_chart(chart_city_bars(ranked, n, highest=True))
    with right:
        show_chart(chart_city_bars(ranked, n, highest=False))
    st.caption(f"City rankings use cities with at least {MIN_READINGS_PER_CITY} AQI readings.")

    spacer()
    render_pipeline_outputs()


def render_explorer(data_path: str | None) -> None:
    page_header("AQI Explorer")
    df = get_data(data_path)
    if df is None:
        return
    pollutants = available_pollutants(df)
    cities = sorted(df["City"].unique())
    buckets = [b for b in BUCKET_ORDER if b in set(df["AQI_Bucket"].dropna())] + sorted(
        b for b in set(df["AQI_Bucket"].dropna()) if b not in BUCKET_ORDER
    )

    section_title("Filters")
    c1, c2 = st.columns([3, 2], gap="medium")
    with c1:
        picked_cities = st.multiselect("Cities (leave empty for all)", cities, key="exp_cities")
    with c2:
        f = apply_date_filter(df, "exp_dates")
    c3, c4, c5, c6 = st.columns([1.2, 1, 2, 1.2], gap="medium")
    with c3:
        metric = st.selectbox("Metric", ["AQI", *pollutants], key="exp_metric")
    with c4:
        freq_label = st.selectbox("Aggregate", list(FREQ), index=2, key="exp_freq")
    with c5:
        picked_buckets = st.multiselect("AQI categories (leave empty for all)", buckets, key="exp_buckets")
    with c6:
        by_city = st.checkbox("One line per city", value=False, key="exp_split")

    if picked_cities:
        f = f[f["City"].isin(picked_cities)]
    if picked_buckets:
        f = f[f["AQI_Bucket"].isin(picked_buckets)]
    if f.empty or f["AQI"].notna().sum() == 0:
        st.info("No records match the current filters.")
        return

    aqi = f["AQI"].dropna()
    peak_row = f.loc[aqi.idxmax()]
    peak_sub = str(peak_row["City"])
    if "Date" in f.columns and pd.notna(peak_row["Date"]):
        peak_sub += f", {peak_row['Date']:%d %b %Y}"
    poor_share = (aqi > POOR_THRESHOLD).mean() * 100
    spacer()
    kpi_row([
        kpi_card("Records", f"{len(f):,}", f"{f['City'].nunique()} cities"),
        kpi_card("Mean AQI", f"{aqi.mean():.0f}", aqi_label(float(aqi.mean()))),
        kpi_card("Median AQI", f"{aqi.median():.0f}", f"std dev {aqi.std():.0f}" if len(aqi) > 1 else ""),
        kpi_card("Peak AQI", f"{aqi.max():.0f}", peak_sub, small=False),
        kpi_card("Poor or worse", f"{poor_share:.1f}%", f"records with AQI above {POOR_THRESHOLD}"),
    ])
    spacer()

    maybe(chart_timeseries(f, metric, freq_label, by_city), "A Date column is needed to draw the time series.")
    spacer()
    left, right = st.columns(2, gap="medium")
    with left:
        maybe(chart_month_pattern(f, metric), "A Date column is needed for the seasonal pattern.")
    with right:
        maybe(chart_histogram(f))

    spacer()
    with st.expander("Filtered data table"):
        show_table(f.head(1000))
        st.caption(f"Showing the first {min(len(f), 1000):,} of {len(f):,} filtered records.")
        st.download_button("Download filtered data (CSV)", f.to_csv(index=False).encode("utf-8"),
                           file_name="airwise_filtered.csv", mime="text/csv")


def render_city_comparison(data_path: str | None) -> None:
    page_header("City Comparison")
    df = get_data(data_path)
    if df is None:
        return
    pollutants = available_pollutants(df)
    ranked = city_stats(df, "AQI")
    cities = sorted(df["City"].unique())
    if len(ranked) >= 4:
        order = ranked["mean"].sort_values(ascending=False).index
        default = [*order[:2], *order[-2:]]
    else:
        default = list(ranked.index)

    section_title("Filters")
    c1, c2, c3 = st.columns([3, 2, 1.2], gap="medium")
    with c1:
        picked = st.multiselect("Cities to compare (2 to 8)", cities, default=default, max_selections=8, key="cmp_cities")
    with c2:
        f = apply_date_filter(df, "cmp_dates")
    with c3:
        metric = st.selectbox("Metric", ["AQI", *pollutants], key="cmp_metric")

    if len(picked) < 2:
        st.info("Select at least two cities to compare.")
        return
    f = f[f["City"].isin(picked)]
    if f.empty:
        st.info("No records match the current filters.")
        return

    spacer()
    left, right = st.columns(2, gap="medium")
    with left:
        maybe(chart_mean_by_city(f, metric))
    with right:
        maybe(chart_box_by_city(f, metric))
    spacer()
    maybe(chart_timeseries(f, metric, "Monthly", by_city=True), "A Date column is needed for the monthly comparison.")

    spacer()
    left, right = st.columns(2, gap="medium")
    with left:
        maybe(chart_category_share(f))
    with right:
        section_title("AQI summary table")
        aqi = f.dropna(subset=["AQI"])
        summary = aqi.groupby("City")["AQI"].agg(Records="count", Mean="mean", Median="median", Max="max", StdDev="std")
        summary["Poor or worse (%)"] = aqi.groupby("City")["AQI"].apply(lambda s: (s > POOR_THRESHOLD).mean() * 100)
        summary = summary.sort_values("Mean", ascending=False).round(1).reset_index()
        show_table(summary)
        st.caption(f"Poor or worse means AQI above {POOR_THRESHOLD}.")


def render_pollutants(data_path: str | None) -> None:
    page_header("Pollutant Analysis")
    df = get_data(data_path)
    if df is None:
        return
    pollutants = available_pollutants(df)
    if not pollutants:
        st.info("No pollutant columns (PM2.5, PM10, NO2, ...) were found in the dataset.")
        return
    cities = sorted(df["City"].unique())

    section_title("Filters")
    c1, c2, c3 = st.columns([3, 2, 1.2], gap="medium")
    with c1:
        picked = st.multiselect("Cities (leave empty for all)", cities, key="pol_cities")
    with c2:
        f = apply_date_filter(df, "pol_dates")
    with c3:
        focus = st.selectbox("Focus pollutant", pollutants, key="pol_focus")
    if picked:
        f = f[f["City"].isin(picked)]
    if f.empty:
        st.info("No records match the current filters.")
        return

    corr = f[["AQI", *pollutants]].corr(numeric_only=True)
    spacer()
    left, right = st.columns(2, gap="medium")
    with left:
        maybe(chart_corr_with_aqi(corr))
    with right:
        show_chart(chart_heatmap(corr))

    spacer()
    left, right = st.columns(2, gap="medium")
    with left:
        scatter, note = chart_scatter(f, focus)
        maybe(scatter)
        if note:
            st.caption(note)
    with right:
        maybe(chart_month_pattern(f, focus), "A Date column is needed for the seasonal pattern.")

    spacer()
    stats = city_stats(f, focus)
    left, right = st.columns(2, gap="medium")
    with left:
        if len(stats) >= 2:
            show_chart(chart_city_bars(stats, min(10, len(stats)), highest=True, metric=focus))
        else:
            st.info("At least two cities are needed for a city ranking.")
    with right:
        show_chart(chart_completeness(f, ["AQI", *pollutants]))
    st.caption("Correlation shows association, not causation. Records with a missing value are skipped for each calculation.")


def render_placeholder(page: str) -> None:
    page_header(page)
    st.markdown(
        f'<div class="panel"><h4>Planned for this page</h4><p>{html.escape(PAGE_PLANS[page])}</p>'
        '<span class="pill">Coming in a later step</span></div>',
        unsafe_allow_html=True,
    )


# --------------------------------------------------------------------------- #
# Sidebar and entry point
# --------------------------------------------------------------------------- #
def render_sidebar(sources: list[str]) -> tuple[str, str | None]:
    data_path: str | None = None
    with st.sidebar:
        st.markdown(
            '<div class="brand"><div class="brand-logo">AW</div><div>'
            '<div class="brand-name">AirWise India</div>'
            '<div class="brand-tag">Distributed AQI analytics on Apache Spark</div>'
            "</div></div>"
            '<div class="nav-label">Navigation</div>',
            unsafe_allow_html=True,
        )
        page = st.radio("Navigation", PAGES, key="page", label_visibility="collapsed")

        st.markdown('<div class="nav-label">Data source</div>', unsafe_allow_html=True)
        if not sources:
            st.markdown(
                f'<div class="side-note">Not found: {html.escape(rel_path(str(PARQUET_PATH)))}</div>',
                unsafe_allow_html=True,
            )
        elif len(sources) == 1:
            data_path = sources[0]
            st.markdown(f'<div class="side-note">{html.escape(rel_path(data_path))}</div>', unsafe_allow_html=True)
        else:
            data_path = st.selectbox("Dataset file", sources, format_func=rel_path, label_visibility="collapsed")

        st.markdown(
            '<div class="side-foot">Streamlit &middot; Plotly &middot; Pandas &middot; PySpark MLlib</div>',
            unsafe_allow_html=True,
        )
    return page, data_path


def main() -> None:
    st.set_page_config(page_title="AirWise India", page_icon="🌫️", layout="wide", initial_sidebar_state="expanded")
    inject_css()
    page, data_path = render_sidebar(resolve_sources())
    renderers = {
        "Overview": render_overview,
        "AQI Explorer": render_explorer,
        "City Comparison": render_city_comparison,
        "Pollutant Analysis": render_pollutants,
    }
    if page in renderers:
        renderers[page](data_path)
    else:
        render_placeholder(page)


main()