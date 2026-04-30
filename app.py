import json
import base64
import io

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Auto EDA", layout="wide", initial_sidebar_state="expanded")


# ── Custom CSS ────────────────────────────────────────────────────────────────

st.markdown("""
<style>
  .stApp { background: #f8f9fb; }
  [data-testid="stSidebar"] { background: #ffffff; border-right: 1px solid #eeeeee; }
  [data-testid="stMetric"] {
    background: #ffffff; border: 1px solid #eeeeee;
    border-radius: 10px; padding: 14px 18px;
  }
  .stTabs [data-baseweb="tab"] { font-size: 13px; font-weight: 500; color: #888; padding: 8px 20px; }
  .stTabs [aria-selected="true"] { color: #1a1a1a; border-bottom: 2px solid #378ADD; }
  .stDownloadButton > button, .stButton > button {
    border-radius: 8px; font-size: 13px; font-weight: 500;
    padding: 8px 20px; background: #378ADD; color: white; border: none;
  }
  .stDownloadButton > button:hover, .stButton > button:hover { background: #2567b8; }
  .streamlit-expanderHeader { font-size: 13px; font-weight: 500; background: #ffffff; border-radius: 8px; }
  #MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ── Constants ─────────────────────────────────────────────────────────────────

NA_VALUES = [
    "", " ", "NA", "N/A", "n/a", "na", "NULL", "null",
    "None", "none", "NaN", "nan", "-", "--", "?", "missing",
    "MISSING", "unknown", "Unknown", "UNK", "#N/A", "#NULL!",
]

ID_HINTS   = ["id", "uuid", "guid", "key", "ref", "index", "no", "num", "number", "code"]
DATE_HINTS = ["date", "time", "datetime", "timestamp", "created", "updated", "at", "on"]
CAT_HINTS  = ["type", "status", "category", "flag", "gender", "sex", "store", "region",
              "country", "zip", "postal", "grade", "class", "label", "group",
              "quarter", "week", "segment", "tier", "level", "rank", "phase"]


# ── LLM Caller ───────────────────────────────────────────────────────────────

def call_llm(prompt: str, provider: str, api_key: str, model: str) -> str:
    if provider == "Gemini":
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        m = genai.GenerativeModel(model)
        return m.generate_content(prompt).text

    elif provider == "OpenAI / ChatGPT":
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        r = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1500,
        )
        return r.choices[0].message.content

    elif provider == "Claude (Anthropic)":
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)
        r = client.messages.create(
            model=model,
            max_tokens=1500,
            messages=[{"role": "user", "content": prompt}],
        )
        return r.content[0].text

    elif provider == "Grok (xAI)":
        from openai import OpenAI  # Grok uses OpenAI-compatible API
        client = OpenAI(api_key=api_key, base_url="https://api.x.ai/v1")
        r = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1500,
        )
        return r.choices[0].message.content

    raise ValueError(f"Unknown provider: {provider}")


# ── Profiler ──────────────────────────────────────────────────────────────────

@st.cache_data
def profile_csv(uploaded_file) -> tuple:
    df = pd.read_csv(uploaded_file, na_values=NA_VALUES, keep_default_na=True)

    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].str.strip().replace("", np.nan)

    for col in df.select_dtypes(include="object").columns:
        sample = df[col].dropna().head(200).astype(str)
        if sample.str.match(r"(\d{4}[-/]\d{2}[-/]\d{2}|\d{2}[-/]\d{2}[-/]\d{4})").mean() > 0.7:
            try:
                df[col] = pd.to_datetime(df[col], infer_datetime_format=True, errors="coerce")
            except Exception:
                pass

    n_rows  = len(df)
    profile = {
        "shape": list(df.shape),
        "columns": {},
        "missing": {c: int(df[c].isnull().sum()) for c in df.columns},
        "duplicates": int(df.duplicated().sum()),
        "memory_mb": round(df.memory_usage(deep=True).sum() / 1e6, 2),
    }

    for col in df.columns:
        n_null       = int(df[col].isnull().sum())
        n_unique     = int(df[col].nunique(dropna=True))
        null_pct     = round(n_null / n_rows * 100, 1)
        col_lower    = col.lower().replace(" ", "_")
        is_date      = pd.api.types.is_datetime64_any_dtype(df[col])
        is_numeric   = pd.api.types.is_numeric_dtype(df[col])
        unique_ratio = n_unique / max(n_rows, 1)

        name_is_id  = any(h == col_lower or col_lower.endswith(f"_{h}") or col_lower.startswith(f"{h}_") for h in ID_HINTS)
        name_is_cat = any(h in col_lower for h in CAT_HINTS)
        looks_like_id   = is_numeric and unique_ratio > 0.95 and n_unique > 100
        low_cardinality = is_numeric and n_unique <= 15
        is_binary       = n_unique == 2

        if is_date:
            parsed = pd.to_datetime(df[col], errors="coerce")
            profile["columns"][col] = {
                "type": "datetime",
                "min": str(parsed.min()), "max": str(parsed.max()),
                "range_days": int((parsed.max() - parsed.min()).days),
                "nulls": n_null, "null_pct": null_pct,
            }
        elif is_numeric and not looks_like_id and not low_cardinality and not name_is_cat and not name_is_id:
            series     = df[col].dropna()
            q1, q3     = series.quantile(0.25), series.quantile(0.75)
            iqr        = q3 - q1
            n_out      = int(((series < q1 - 1.5*iqr) | (series > q3 + 1.5*iqr)).sum())
            skewness   = round(float(series.skew()), 3)
            skew_label = "right-skewed" if skewness > 1 else "left-skewed" if skewness < -1 else "symmetric"
            profile["columns"][col] = {
                "type": "numeric",
                "stats": series.describe().round(4).to_dict(),
                "skewness": skewness, "skew_label": skew_label,
                "outliers": n_out,
                "outlier_pct": round(n_out / max(len(series), 1) * 100, 1),
                "nulls": n_null, "null_pct": null_pct,
            }
        elif is_binary:
            vals = df[col].value_counts(normalize=True).round(3).to_dict()
            profile["columns"][col] = {
                "type": "binary",
                "values": {str(k): float(v) for k, v in vals.items()},
                "nulls": n_null, "null_pct": null_pct,
            }
        else:
            profile["columns"][col] = {
                "type": "id" if (looks_like_id or name_is_id) else "categorical",
                "unique": n_unique,
                "unique_pct": round(unique_ratio * 100, 1),
                "top5": df[col].astype(str).value_counts().head(5).to_dict(),
                "nulls": n_null, "null_pct": null_pct,
            }

    return df, profile


# ── Target Analysis ───────────────────────────────────────────────────────────

def analyze_target(df, profile, target_col):
    meta    = profile["columns"].get(target_col, {})
    t_type  = meta.get("type", "categorical")
    charts  = []
    insights = []

    task = "classification" if t_type in ("binary", "categorical") or \
           (t_type == "numeric" and df[target_col].nunique() <= 15) else "regression"

    if task == "classification":
        vc = df[target_col].value_counts().reset_index()
        vc.columns = [target_col, "count"]
        vc["pct"] = (vc["count"] / len(df) * 100).round(1)
        fig = px.bar(vc, x=target_col, y="count",
                     text=vc["pct"].astype(str) + "%",
                     color_discrete_sequence=["#378ADD"],
                     title=f"Target distribution: {target_col}")
        fig.update_layout(**clean_layout())
        charts.append(("target_dist", target_col, fig))
        ratios = vc["count"] / vc["count"].sum()
        if ratios.min() < 0.1:
            insights.append(f"⚠️ **Class imbalance** — minority class is only {round(ratios.min()*100,1)}%. Consider SMOTE or class weighting.")
        else:
            insights.append("✅ Classes are reasonably balanced.")
    else:
        fig = px.histogram(df[target_col].dropna(), nbins=40,
                           color_discrete_sequence=["#7F77DD"],
                           title=f"Target distribution: {target_col}")
        fig.update_layout(**clean_layout())
        charts.append(("target_dist", target_col, fig))
        skew = round(float(df[target_col].skew()), 3)
        if abs(skew) > 1:
            insights.append(f"⚠️ **Target is skewed** ({skew}) — consider log-transforming `{target_col}`.")
        else:
            insights.append(f"✅ Target looks reasonable (skewness: {skew}).")

    num_cols = [c for c, m in profile["columns"].items() if m["type"] == "numeric" and c != target_col]
    cat_cols = [c for c, m in profile["columns"].items() if m["type"] in ("categorical", "binary") and c != target_col]

    if num_cols:
        try:
            corr_vals = df[num_cols + [target_col]].corr()[target_col].drop(target_col).abs().sort_values(ascending=False)
            top_corr  = corr_vals.head(10).reset_index()
            top_corr.columns = ["feature", "correlation"]
            fig = px.bar(top_corr, x="correlation", y="feature", orientation="h",
                         color_discrete_sequence=["#1D9E75"],
                         title=f"Feature correlation with {target_col}")
            fig.update_layout(**clean_layout())
            charts.append(("feature_corr", "correlation", fig))
            if len(corr_vals) > 0:
                top_feat = corr_vals.index[0]
                insights.append(f"🔵 **Strongest predictor:** `{top_feat}` (correlation: {round(corr_vals[top_feat], 3)})")
        except Exception:
            pass

    for col in cat_cols[:3]:
        try:
            if task == "regression":
                grp = df.groupby(col)[target_col].mean().reset_index().sort_values(target_col, ascending=False).head(10)
                fig = px.bar(grp, x=target_col, y=col, orientation="h",
                             color_discrete_sequence=["#E8963A"],
                             title=f"Mean {target_col} by {col}")
            else:
                grp = df.groupby([col, target_col]).size().reset_index(name="count")
                fig = px.bar(grp, x=col, y="count", color=str(target_col),
                             barmode="group", title=f"{col} vs {target_col}")
            fig.update_layout(**clean_layout())
            charts.append(("cat_vs_target", col, fig))
        except Exception:
            pass

    return task, charts, insights


# ── Auto Clean ────────────────────────────────────────────────────────────────

def auto_clean(df, profile, target_col=None):
    cleaned = df.copy()
    log     = []

    id_cols = [c for c, m in profile["columns"].items() if m["type"] == "id"]
    if id_cols:
        cleaned.drop(columns=id_cols, inplace=True, errors="ignore")
        log.append(f"🗑 Dropped ID columns: {', '.join(id_cols)}")

    high_missing = [c for c, m in profile["columns"].items()
                    if m["null_pct"] > 60 and c != target_col]
    if high_missing:
        cleaned.drop(columns=high_missing, inplace=True, errors="ignore")
        log.append(f"🗑 Dropped high-missing columns (>60%): {', '.join(high_missing)}")

    n_before = len(cleaned)
    cleaned.drop_duplicates(inplace=True)
    if len(cleaned) < n_before:
        log.append(f"🗑 Dropped {n_before - len(cleaned)} duplicate rows")

    for col in [c for c in cleaned.columns
                if c in profile["columns"] and profile["columns"][c]["type"] == "numeric"
                and cleaned[c].isnull().any()]:
        median = cleaned[col].median()
        cleaned[col].fillna(median, inplace=True)
        log.append(f"🔧 Imputed `{col}` with median ({round(median, 3)})")

    for col in [c for c in cleaned.columns
                if c in profile["columns"] and profile["columns"][c]["type"] in ("categorical", "binary")
                and cleaned[c].isnull().any()]:
        mode = cleaned[col].mode()[0]
        cleaned[col].fillna(mode, inplace=True)
        log.append(f"🔧 Imputed `{col}` with mode ('{mode}')")

    for col in [c for c in cleaned.columns
                if c in profile["columns"]
                and profile["columns"][c]["type"] == "numeric"
                and profile["columns"][c]["skewness"] > 1
                and (cleaned[c] > 0).all()
                and c != target_col]:
        cleaned[f"{col}_log"] = np.log1p(cleaned[col])
        log.append(f"📐 Created `{col}_log` (log-transform)")

    for col in [c for c in cleaned.columns
                if c in profile["columns"] and profile["columns"][c]["type"] == "binary"
                and c != target_col]:
        vals = cleaned[col].dropna().unique()
        if set(vals) != {0, 1}:
            cleaned[col] = (cleaned[col] == vals[0]).astype(int)
            log.append(f"🔢 Encoded `{col}` as 0/1")

    for col in [c for c in cleaned.columns
                if c in profile["columns"] and profile["columns"][c]["type"] == "datetime"]:
        try:
            parsed = pd.to_datetime(cleaned[col], errors="coerce")
            cleaned[f"{col}_year"]  = parsed.dt.year
            cleaned[f"{col}_month"] = parsed.dt.month
            cleaned[f"{col}_dow"]   = parsed.dt.dayofweek
            cleaned.drop(columns=[col], inplace=True)
            log.append(f"📅 Extracted year/month/dow from `{col}`")
        except Exception:
            pass

    return cleaned, log


# ── Charts ────────────────────────────────────────────────────────────────────

def clean_layout(height=280):
    return dict(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="system-ui, sans-serif", size=12, color="#555"),
        title=dict(font=dict(size=13), x=0),
        margin=dict(l=0, r=0, t=36, b=0),
        xaxis=dict(showgrid=True, gridcolor="#f0f0f0", zeroline=False),
        yaxis=dict(showgrid=False, zeroline=False),
        showlegend=False,
        height=height,
    )


@st.cache_data
def generate_charts(_df, profile):
    charts   = []
    num_cols  = [c for c, m in profile["columns"].items() if m["type"] == "numeric"]
    cat_cols  = [c for c, m in profile["columns"].items() if m["type"] == "categorical"]
    bin_cols  = [c for c, m in profile["columns"].items() if m["type"] == "binary"]
    date_cols = [c for c, m in profile["columns"].items() if m["type"] == "datetime"]

    for col in num_cols:
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=_df[col].dropna(), nbinsx=30,
                                   marker_color="#378ADD", opacity=0.85, name=col))
        fig.update_layout(**clean_layout(), title=f"Distribution: {col}")
        charts.append(("numeric", col, fig))

        fig2 = go.Figure()
        fig2.add_trace(go.Box(x=_df[col].dropna(), marker_color="#378ADD", boxmean=True, name=col))
        fig2.update_layout(**clean_layout(height=200), title=f"Boxplot: {col}")
        charts.append(("boxplot", col, fig2))

    for col in cat_cols:
        top = _df[col].value_counts().head(12).reset_index()
        top.columns = [col, "count"]
        fig = px.bar(top, x="count", y=col, orientation="h",
                     color_discrete_sequence=["#1D9E75"], title=f"Top values: {col}")
        fig.update_layout(**clean_layout())
        fig.update_traces(marker_line_width=0)
        charts.append(("categorical", col, fig))

    for col in bin_cols:
        vals = profile["columns"][col]["values"]
        fig  = go.Figure(go.Pie(labels=list(vals.keys()), values=list(vals.values()),
                                hole=0.55, marker_colors=["#378ADD", "#E8F3FB"]))
        fig.update_layout(**clean_layout(height=240), title=f"Split: {col}")
        charts.append(("binary", col, fig))

    for col in date_cols:
        try:
            series   = pd.to_datetime(_df[col], errors="coerce").dropna()
            span     = (series.max() - series.min()).days
            freq     = "D" if span <= 90 else "W" if span <= 365 else "M"
            timeline = series.dt.to_period(freq).astype(str).value_counts().sort_index().reset_index()
            timeline.columns = ["period", "count"]
            fig = px.line(timeline, x="period", y="count",
                          color_discrete_sequence=["#7F77DD"], title=f"Over time: {col}")
            fig.update_layout(**clean_layout())
            charts.append(("datetime", col, fig))
        except Exception:
            pass

    missing_cols = [c for c in _df.columns if _df[c].isnull().any()]
    if missing_cols:
        sample = _df[missing_cols].isnull().astype(int).head(200)
        fig = px.imshow(sample.T, color_continuous_scale=["#f0f0f0", "#E05A3A"],
                        title="Missing value map (sample of 200 rows)", aspect="auto")
        fig.update_layout(**clean_layout(height=max(200, len(missing_cols) * 20 + 60)))
        fig.update_coloraxes(showscale=False)
        charts.append(("missing", "missing_map", fig))

    if len(num_cols) > 1:
        corr = _df[num_cols].corr().round(2)
        fig  = px.imshow(corr, text_auto=True, color_continuous_scale="Blues",
                         title="Correlation matrix", aspect="auto")
        fig.update_layout(**clean_layout(height=max(300, len(num_cols) * 40 + 80)))
        charts.append(("corr", "correlation", fig))

    return charts


# ── LLM Summary ───────────────────────────────────────────────────────────────

@st.cache_data
def generate_summary(profile, target_col, provider, api_key, model):
    target = f"The user has selected `{target_col}` as the target variable." if target_col else "No target variable selected."
    prompt = f"""You are a senior data scientist writing an EDA report.

{target}

Dataset profile:
{json.dumps(profile, indent=2)}

Write a concise but thorough EDA report using markdown with these sections:

## Overview
Shape, memory, column type breakdown, overall data quality score (0-100).

## Data Quality
Missing values per column with recommended imputation strategy. Duplicates. Suspicious columns.

## Key Patterns
Numeric distributions and skewness. Categorical breakdowns. Datetime trends. Binary splits.

## Target Variable Analysis
(Only if target provided) Task type, class balance, strongest predictors, risks.

## Feature Engineering Suggestions
4-5 concrete suggestions with actual column names.

## Modelling Readiness
What still needs doing. Leakage risks. Recommended algorithm family.

Mention actual column names. Be specific. Under 700 words."""

    return call_llm(prompt, provider, api_key, model)


# ── HTML Report ───────────────────────────────────────────────────────────────

def fig_to_b64(fig):
    buf = io.BytesIO()
    fig.write_image(buf, format="png", scale=2)
    return base64.b64encode(buf.getvalue()).decode()


def build_html_report(df, profile, summary, charts, target_col=None):
    charts_html   = ""
    for i in range(0, len(charts), 2):
        charts_html += '<div style="display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-bottom:16px">'
        for _, col, fig in charts[i:i+2]:
            try:
                charts_html += f'<img src="data:image/png;base64,{fig_to_b64(fig)}" style="width:100%;border-radius:8px;border:1px solid #f0f0f0">'
            except Exception:
                pass
        charts_html += "</div>"

    missing_total = sum(profile["missing"].values())
    target_line   = f"&nbsp;·&nbsp; Target: <strong>{target_col}</strong>" if target_col else ""

    html = f"""<!DOCTYPE html>
<html><head><meta charset="UTF-8">
<style>
  body {{ font-family: system-ui, sans-serif; max-width: 960px; margin: 40px auto; padding: 0 28px; color: #1a1a1a; line-height: 1.7; }}
  h1 {{ font-size: 22px; font-weight: 500; margin-bottom: 4px; }}
  h2 {{ font-size: 16px; font-weight: 500; border-bottom: 1px solid #eee; padding-bottom: 6px; margin-top: 28px; }}
  .meta {{ font-size: 13px; color: #888; margin-bottom: 20px; }}
  .metrics {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin-bottom: 24px; }}
  .metric {{ background: #f7f7f7; border-radius: 8px; padding: 14px 16px; }}
  .metric-val {{ font-size: 26px; font-weight: 500; margin: 0; }}
  .metric-lbl {{ font-size: 12px; color: #888; margin: 4px 0 0; }}
  .summary {{ font-size: 14px; background: #f9f9f9; padding: 20px 24px; border-radius: 8px; }}
  ul {{ margin: 6px 0; padding-left: 20px; }}
  li {{ margin-bottom: 4px; font-size: 14px; }}
  code {{ background: #f0f0f0; padding: 1px 5px; border-radius: 3px; font-size: 13px; }}
</style>
</head><body>
<h1>EDA Report</h1>
<div class="meta">{df.shape[0]:,} rows &nbsp;·&nbsp; {df.shape[1]} columns &nbsp;·&nbsp; {profile.get("memory_mb","?")} MB{target_line}</div>
<div class="metrics">
  <div class="metric"><p class="metric-val">{df.shape[0]:,}</p><p class="metric-lbl">Rows</p></div>
  <div class="metric"><p class="metric-val">{df.shape[1]}</p><p class="metric-lbl">Columns</p></div>
  <div class="metric"><p class="metric-val">{missing_total:,}</p><p class="metric-lbl">Missing values</p></div>
  <div class="metric"><p class="metric-val">{profile["duplicates"]}</p><p class="metric-lbl">Duplicates</p></div>
</div>
<h2>AI Analysis</h2>
<div class="summary">{summary.replace(chr(10), "<br>")}</div>
<h2>Visualizations</h2>
{charts_html}
</body></html>"""

    return html


# ── Sidebar ───────────────────────────────────────────────────────────────────

PROVIDERS = {
    "Gemini":            {"models": ["gemini-1.5-flash", "gemini-1.5-pro", "gemini-2.0-flash"], "key_hint": "AIza..."},
    "OpenAI / ChatGPT":  {"models": ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"],                "key_hint": "sk-..."},
    "Claude (Anthropic)":{"models": ["claude-haiku-4-5-20251001", "claude-sonnet-4-6"],         "key_hint": "sk-ant-..."},
    "Grok (xAI)":        {"models": ["grok-3-mini", "grok-3"],                                  "key_hint": "xai-..."},
}

with st.sidebar:
    st.markdown("### ⚙️ Settings")
    st.divider()

    uploaded = st.file_uploader("Upload CSV", type=["csv"], label_visibility="collapsed")

    target_col = None
    do_clean   = False

    if uploaded:
        st.markdown("**Target variable**")
        df_peek = pd.read_csv(uploaded, nrows=5)
        uploaded.seek(0)
        col_options = ["— none —"] + list(df_peek.columns)
        target_sel  = st.selectbox("Target column", col_options, label_visibility="collapsed")
        target_col  = None if target_sel == "— none —" else target_sel

        st.markdown("**Auto-clean**")
        do_clean = st.toggle("Clean & prepare dataset", value=False)
        if do_clean:
            st.caption("Drops IDs, imputes missing, encodes binary, extracts dates, log-transforms skewed columns.")

    st.divider()
    st.markdown("**AI Provider**")
    provider = st.selectbox("Provider", list(PROVIDERS.keys()), label_visibility="collapsed")
    model    = st.selectbox("Model", PROVIDERS[provider]["models"], label_visibility="collapsed")
    api_key  = st.text_input(
        "API Key",
        type="password",
        placeholder=PROVIDERS[provider]["key_hint"],
        label_visibility="collapsed",
    )

    key_links = {
        "Gemini":             "https://aistudio.google.com",
        "OpenAI / ChatGPT":   "https://platform.openai.com/api-keys",
        "Claude (Anthropic)": "https://console.anthropic.com",
        "Grok (xAI)":         "https://console.x.ai",
    }
    st.caption(f"Get a free key → [{key_links[provider]}]({key_links[provider]})")

    st.divider()
    st.caption("🔵 numeric  🟢 categorical\n🟣 datetime  🟡 binary  ⚫ id")


# ── Main ──────────────────────────────────────────────────────────────────────

if not uploaded:
    st.markdown("""
    <div style="text-align:center;padding:80px 0 40px">
      <div style="font-size:40px;margin-bottom:16px">📊</div>
      <div style="font-size:20px;font-weight:500;margin-bottom:8px">Auto EDA Tool</div>
      <div style="font-size:14px;color:#888;margin-bottom:4px">Upload a CSV in the sidebar to get started</div>
      <div style="font-size:13px;color:#aaa">Works with any AI provider — Gemini, ChatGPT, Claude, Grok</div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

with st.spinner("Profiling dataset..."):
    df, profile = profile_csv(uploaded)

m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Rows",       f"{profile['shape'][0]:,}")
m2.metric("Columns",    profile["shape"][1])
m3.metric("Missing",    f"{sum(profile['missing'].values()):,}")
m4.metric("Duplicates", profile["duplicates"])
m5.metric("Memory",     f"{profile.get('memory_mb','?')} MB")

if target_col:
    st.info(f"🎯 Target variable: **{target_col}**")

st.divider()

tab1, tab2, tab3, tab4, tab5 = st.tabs(["Columns", "Charts", "Target Analysis", "Auto-Clean", "AI Summary + Report"])


# ── Tab 1: Columns ────────────────────────────────────────────────────────────
with tab1:
    type_colors = {"numeric": "🔵", "categorical": "🟢", "datetime": "🟣", "binary": "🟡", "id": "⚫"}
    for col, meta in profile["columns"].items():
        t    = meta["type"]
        icon = type_colors.get(t, "⚪")
        warn = f"  ⚠ {meta['nulls']} missing ({meta['null_pct']}%)" if meta["nulls"] > 0 else ""
        tag  = " 🎯" if col == target_col else ""
        with st.expander(f"{icon} **{col}**{tag}  ·  {t}{warn}"):
            if t == "numeric":
                c1, c2, c3 = st.columns(3)
                c1.metric("Mean",     round(meta["stats"]["mean"], 3))
                c2.metric("Std",      round(meta["stats"]["std"], 3))
                c3.metric("Outliers", f"{meta['outliers']} ({meta['outlier_pct']}%)")
                st.dataframe(pd.DataFrame(meta["stats"], index=["value"]).T, use_container_width=True)
                if meta["outliers"] > 0:
                    st.warning(f"Skewness: {meta['skewness']} ({meta['skew_label']}) — {meta['outliers']} outliers")
            elif t == "datetime":
                st.write(f"**Range:** {meta['min']}  →  {meta['max']}")
                st.write(f"**Span:** {meta['range_days']} days")
            elif t == "binary":
                for val, pct in meta["values"].items():
                    st.write(f"`{val}` — {round(pct*100,1)}%")
            elif t == "id":
                st.info(f"Likely an ID column ({meta['unique']} unique, {meta['unique_pct']}% of rows). Will be dropped in auto-clean.")
            else:
                st.write(f"**Unique:** {meta['unique']} ({meta['unique_pct']}% of rows)")
                st.dataframe(pd.DataFrame.from_dict(meta["top5"], orient="index", columns=["count"]), use_container_width=True)
            if meta["nulls"] > 0:
                st.warning(f"{meta['nulls']} missing values ({meta['null_pct']}% of rows)")


# ── Tab 2: Charts ─────────────────────────────────────────────────────────────
with tab2:
    with st.spinner("Generating charts..."):
        charts = generate_charts(df, profile)
    sections = {
        "numeric": "Numeric distributions", "boxplot": "Boxplots",
        "categorical": "Categorical breakdowns", "binary": "Binary splits",
        "datetime": "Time series", "missing": "Missing value map",
        "corr": "Correlation matrix",
    }
    for type_key, section_title in sections.items():
        section_charts = [(k, c, f) for k, c, f in charts if k == type_key]
        if not section_charts:
            continue
        st.subheader(section_title)
        left, right = st.columns(2)
        for i, (_, col, fig) in enumerate(section_charts):
            (left if i % 2 == 0 else right).plotly_chart(fig, use_container_width=True)


# ── Tab 3: Target Analysis ────────────────────────────────────────────────────
with tab3:
    if not target_col:
        st.info("👈 Select a target variable in the sidebar to see target analysis.")
    else:
        with st.spinner(f"Analysing target: {target_col}..."):
            task, t_charts, insights = analyze_target(df, profile, target_col)
        st.markdown(f"**Task type detected:** `{task}`")
        for ins in insights:
            st.markdown(ins)
        st.divider()
        left, right = st.columns(2)
        for i, (_, col, fig) in enumerate(t_charts):
            (left if i % 2 == 0 else right).plotly_chart(fig, use_container_width=True)


# ── Tab 4: Auto-Clean ─────────────────────────────────────────────────────────
with tab4:
    if not do_clean:
        st.info("👈 Toggle **Clean & prepare dataset** in the sidebar to use this feature.")
    else:
        with st.spinner("Cleaning dataset..."):
            cleaned_df, clean_log = auto_clean(df, profile, target_col)
        st.success(f"Done — {df.shape[1]} cols, {len(df):,} rows → {cleaned_df.shape[1]} cols, {len(cleaned_df):,} rows")
        for entry in clean_log:
            st.markdown(entry)
        st.divider()
        st.subheader("Preview")
        st.dataframe(cleaned_df.head(20), use_container_width=True)
        st.download_button(
            label="Download cleaned CSV",
            data=cleaned_df.to_csv(index=False).encode("utf-8"),
            file_name="cleaned_data.csv",
            mime="text/csv",
        )


# ── Tab 5: AI Summary + Report ────────────────────────────────────────────────
with tab5:
    if not api_key:
        st.info(f"👈 Enter your {provider} API key in the sidebar to generate the AI summary.")
    else:
        with st.spinner(f"Generating analysis with {provider} / {model}..."):
            try:
                summary = generate_summary(profile, target_col, provider, api_key, model)
                st.markdown(summary)
                st.divider()
                with st.spinner("Building report..."):
                    html = build_html_report(df, profile, summary, charts, target_col)
                st.download_button(
                    label="Download full report (.html)",
                    data=html,
                    file_name="eda_report.html",
                    mime="text/html",
                )
            except Exception as e:
                st.error(f"Error: {e}")
