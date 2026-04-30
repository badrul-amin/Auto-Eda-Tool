import json
import base64
import io

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Auto EDA", layout="wide", initial_sidebar_state="expanded")


# ── CSS ───────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
  .stApp { background: #f4f6f9; }
  .block-container { padding-top: 1.5rem !important; }
  [data-testid="stSidebar"] { background: #ffffff; border-right: 1px solid #e8ecf0; }
  [data-testid="stMetric"] { background: #ffffff; border: 1px solid #e8ecf0; border-radius: 12px; padding: 16px 18px !important; }
  [data-testid="stMetricLabel"] { font-size: 12px !important; color: #888 !important; }
  [data-testid="stMetricValue"] { font-size: 24px !important; font-weight: 500 !important; }
  .stTabs [data-baseweb="tab-list"] { background: #ffffff; border-radius: 12px; padding: 4px; border: 1px solid #e8ecf0; gap: 2px; }
  .stTabs [data-baseweb="tab"] { font-size: 13px; font-weight: 500; color: #888; padding: 8px 18px; border-radius: 8px; }
  .stTabs [aria-selected="true"] { background: #378ADD !important; color: white !important; }
  .stButton > button { border-radius: 8px; font-size: 13px; font-weight: 500; padding: 8px 20px; background: #378ADD; color: white !important; border: none; }
  .stButton > button:hover { background: #2567b8 !important; }
  .stDownloadButton > button { border-radius: 8px; font-size: 13px; font-weight: 500; padding: 10px 20px; background: #1D9E75; color: white !important; border: none; }
  .stDownloadButton > button:hover { background: #157a5a !important; }
  .streamlit-expanderHeader { background: #ffffff !important; border-radius: 10px !important; font-size: 13px; font-weight: 500; }
  #MainMenu, footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────────────────

NA_VALUES = [
    "", " ", "NA", "N/A", "n/a", "na", "NULL", "null",
    "None", "none", "NaN", "nan", "-", "--", "?",
    "missing", "MISSING", "unknown", "Unknown", "UNK", "#N/A", "#NULL!",
]
ID_HINTS = ["id", "uuid", "guid", "key", "ref", "index", "no", "num", "number", "code", "item", "sku", "barcode"]
DATE_HINTS = ["date", "time", "datetime", "timestamp", "created", "updated", "at", "on"]
CAT_HINTS  = ["type", "status", "category", "flag", "gender", "sex", "store", "region",
              "country", "zip", "postal", "grade", "class", "label", "group",
              "quarter", "week", "segment", "tier", "level", "rank", "phase"]

PROVIDERS = {
    "Gemini":             {"models": ["gemini-1.5-flash", "gemini-1.5-pro", "gemini-2.0-flash"], "key_hint": "AIza..."},
    "OpenAI / ChatGPT":   {"models": ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"],                 "key_hint": "sk-..."},
    "Claude (Anthropic)": {"models": ["claude-haiku-4-5-20251001", "claude-sonnet-4-6"],          "key_hint": "sk-ant-..."},
    "Grok (xAI)":         {"models": ["grok-3-mini", "grok-3"],                                   "key_hint": "xai-..."},
}


# ── Session state ─────────────────────────────────────────────────────────────

for key, val in {"provider": "Gemini", "model": "gemini-1.5-flash", "api_key": "", "step": 1}.items():
    if key not in st.session_state:
        st.session_state[key] = val


# ── LLM caller ────────────────────────────────────────────────────────────────

def call_llm(prompt, provider, api_key, model):
    if provider == "Gemini":
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        return genai.GenerativeModel(model).generate_content(prompt).text

    elif provider == "OpenAI / ChatGPT":
        from openai import OpenAI
        r = OpenAI(api_key=api_key).chat.completions.create(
            model=model, messages=[{"role": "user", "content": prompt}], max_tokens=1500)
        return r.choices[0].message.content

    elif provider == "Claude (Anthropic)":
        import anthropic
        r = anthropic.Anthropic(api_key=api_key).messages.create(
            model=model, max_tokens=1500,
            messages=[{"role": "user", "content": prompt}])
        return r.content[0].text

    elif provider == "Grok (xAI)":
        from openai import OpenAI
        r = OpenAI(api_key=api_key, base_url="https://api.x.ai/v1").chat.completions.create(
            model=model, messages=[{"role": "user", "content": prompt}], max_tokens=1500)
        return r.choices[0].message.content

    raise ValueError(f"Unknown provider: {provider}")


# ── Profiler ──────────────────────────────────────────────────────────────────

@st.cache_data
def profile_csv(uploaded_file):
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
        "shape":      list(df.shape),
        "columns":    {},
        "missing":    {c: int(df[c].isnull().sum()) for c in df.columns},
        "duplicates": int(df.duplicated().sum()),
        "memory_mb":  round(df.memory_usage(deep=True).sum() / 1e6, 2),
    }

    for col in df.columns:
        n_null       = int(df[col].isnull().sum())
        n_unique     = int(df[col].nunique(dropna=True))
        null_pct     = round(n_null / n_rows * 100, 1)
        col_lower    = col.lower().replace(" ", "_")
        is_date      = pd.api.types.is_datetime64_any_dtype(df[col])
        is_numeric   = pd.api.types.is_numeric_dtype(df[col])
        unique_ratio = n_unique / max(n_rows, 1)

        name_is_id  = any(
          h == col_lower or
          col_lower.endswith(f"_{h}") or
          col_lower.startswith(f"{h}_") or
          f"_{h}" in col_lower or
          f"{h}_" in col_lower or
          h in col_lower.split("_")
          for h in ID_HINTS
        )
        name_is_cat     = any(h in col_lower for h in CAT_HINTS)
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
            n_out      = int(((series < q1 - 1.5 * iqr) | (series > q3 + 1.5 * iqr)).sum())
            skewness   = round(float(series.skew()), 3)
            skew_label = "right-skewed" if skewness > 1 else "left-skewed" if skewness < -1 else "symmetric"
            profile["columns"][col] = {
                "type": "numeric", "stats": series.describe().round(4).to_dict(),
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
                "unique": n_unique, "unique_pct": round(unique_ratio * 100, 1),
                "top5": df[col].astype(str).value_counts().head(5).to_dict(),
                "nulls": n_null, "null_pct": null_pct,
            }

    return df, profile


# ── Charts ────────────────────────────────────────────────────────────────────

def clean_layout(height=280):
    return dict(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="system-ui, sans-serif", size=12, color="#555"),
        title=dict(font=dict(size=13), x=0),
        margin=dict(l=0, r=0, t=36, b=0),
        xaxis=dict(showgrid=True, gridcolor="#f0f0f0", zeroline=False),
        yaxis=dict(showgrid=False, zeroline=False),
        showlegend=False, height=height,
    )


@st.cache_data
def generate_charts(_df, profile):
    charts    = []
    num_cols  = [c for c, m in profile["columns"].items() if m["type"] == "numeric"]
    cat_cols  = [c for c, m in profile["columns"].items() if m["type"] == "categorical"]
    bin_cols  = [c for c, m in profile["columns"].items() if m["type"] == "binary"]
    date_cols = [c for c, m in profile["columns"].items() if m["type"] == "datetime"]

    for col in num_cols:
        # Histogram
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=_df[col].dropna(), nbinsx=30,
            marker_color="#378ADD", opacity=0.85, name=col,
        ))
        layout = clean_layout()
        layout["title"] = {"text": f"Distribution: {col}", "font": {"size": 13}, "x": 0}
        fig.update_layout(**layout)
        charts.append(("numeric", col, fig))

        # Boxplot
        fig2 = go.Figure()
        fig2.add_trace(go.Box(
            x=_df[col].dropna(), marker_color="#378ADD", boxmean=True, name=col,
        ))
        layout2 = clean_layout(height=200)
        layout2["title"] = {"text": f"Boxplot: {col}", "font": {"size": 13}, "x": 0}
        fig2.update_layout(**layout2)
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
        fig  = go.Figure(go.Pie(
            labels=list(vals.keys()), values=list(vals.values()),
            hole=0.55, marker_colors=["#378ADD", "#E8F3FB"],
        ))
        layout3 = clean_layout(height=240)
        layout3["title"] = {"text": f"Split: {col}", "font": {"size": 13}, "x": 0}
        fig.update_layout(**layout3)
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


# ── Target Analysis ───────────────────────────────────────────────────────────

def analyze_target(df, profile, target_col):
    meta     = profile["columns"].get(target_col, {})
    t_type   = meta.get("type", "categorical")
    charts   = []
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
            insights.append(f"⚠️ **Class imbalance** — minority class is {round(ratios.min()*100,1)}%. Consider SMOTE or class weighting.")
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
            insights.append(f"⚠️ Target is skewed ({skew}). Consider log-transform.")
        else:
            insights.append(f"✅ Target distribution looks reasonable (skewness: {skew}).")

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
                insights.append(f"🔵 **Strongest predictor:** `{top_feat}` (r = {round(corr_vals[top_feat], 3)})")
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

    high_miss = [c for c, m in profile["columns"].items() if m["null_pct"] > 60 and c != target_col]
    if high_miss:
        cleaned.drop(columns=high_miss, inplace=True, errors="ignore")
        log.append(f"🗑 Dropped columns with >60% missing: {', '.join(high_miss)}")

    n_before = len(cleaned)
    cleaned.drop_duplicates(inplace=True)
    if len(cleaned) < n_before:
        log.append(f"🗑 Dropped {n_before - len(cleaned)} duplicate rows")

    for col in [c for c in cleaned.columns if c in profile["columns"]
                and profile["columns"][c]["type"] == "numeric"
                and cleaned[c].isnull().any()]:
        med = cleaned[col].median()
        cleaned[col].fillna(med, inplace=True)
        log.append(f"🔧 Imputed `{col}` with median ({round(med, 3)})")

    for col in [c for c in cleaned.columns if c in profile["columns"]
                and profile["columns"][c]["type"] in ("categorical", "binary")
                and cleaned[c].isnull().any()]:
        mode = cleaned[col].mode()[0]
        cleaned[col].fillna(mode, inplace=True)
        log.append(f"🔧 Imputed `{col}` with mode ('{mode}')")

    for col in [c for c in cleaned.columns if c in profile["columns"]
                and profile["columns"][c]["type"] == "numeric"
                and profile["columns"][c]["skewness"] > 1
                and (cleaned[c] > 0).all() and c != target_col]:
        cleaned[f"{col}_log"] = np.log1p(cleaned[col])
        log.append(f"📐 Log-transformed `{col}` → `{col}_log`")

    for col in [c for c in cleaned.columns if c in profile["columns"]
                and profile["columns"][c]["type"] == "binary" and c != target_col]:
        vals = cleaned[col].dropna().unique()
        if set(vals) != {0, 1}:
            cleaned[col] = (cleaned[col] == vals[0]).astype(int)
            log.append(f"🔢 Encoded `{col}` as 0/1")

    for col in [c for c in cleaned.columns if c in profile["columns"]
                and profile["columns"][c]["type"] == "datetime"]:
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


# ── LLM Summary ───────────────────────────────────────────────────────────────

@st.cache_data
def generate_summary(profile, target_col, provider, api_key, model, _df):
    target = f"The user selected `{target_col}` as the target variable." if target_col else "No target variable selected."

    # Send actual sample rows so the LLM can see real values
    sample_rows = _df.head(10).to_string()
    sample_stats = _df.describe(include="all").round(2).to_string()

    # Build a smarter context
    col_types = "\n".join(
        f"- `{col}`: {meta['type']}  |  {meta['nulls']} nulls ({meta['null_pct']}%)"
        + (f"  |  skewness: {meta.get('skewness','N/A')}  |  outliers: {meta.get('outliers','N/A')}" if meta['type'] == 'numeric' else "")
        + (f"  |  top values: {list(meta.get('top5',{}).keys())}" if meta['type'] in ('categorical','id') else "")
        for col, meta in profile["columns"].items()
    )

    prompt = f"""You are a senior data scientist doing exploratory data analysis.

{target}

DATASET SHAPE: {profile['shape'][0]:,} rows × {profile['shape'][1]} columns
MEMORY: {profile.get('memory_mb','?')} MB
DUPLICATES: {profile['duplicates']}

COLUMN SUMMARY:
{col_types}

ACTUAL DATA SAMPLE (first 10 rows):
{sample_rows}

DESCRIPTIVE STATISTICS:
{sample_stats}

Based on the ACTUAL data above, write a thorough EDA report in markdown:

## Overview
Dataset shape, what this data appears to be about (guess from column names and values), data quality score (0-100).

## Data Quality Issues
- List every column with missing values, how many, and exact recommended fix
- Flag any columns with suspicious values, outliers, or wrong types
- Note duplicates

## Key Patterns & Insights
- What do the actual values tell you? Look at the real sample data
- Numeric columns: distribution shape, notable outliers, value ranges
- Categorical columns: dominant categories, rare categories, imbalance
- Any obvious business insights from the data (e.g. if it's sales data, comment on sales patterns)

## Target Variable Analysis
Only if target is provided: task type, class balance, which features look most predictive and why.

## Feature Engineering Suggestions
5 concrete suggestions referencing actual column names. Be specific — e.g. "extract month from `Order Date` to capture seasonality".

## Modelling Readiness
Exact steps needed before training. Call out leakage risks by column name. Recommend specific algorithm.

## Anomalies & Watch-outs
Anything unusual spotted in the actual data sample — weird values, impossible combinations, potential data entry errors.

Be specific. Reference actual values you see in the data. Think like a data scientist who just opened this file in pandas."""

    return call_llm(prompt, provider, api_key, model)

# ── HTML Report ───────────────────────────────────────────────────────────────

def fig_to_b64(fig):
    buf = io.BytesIO()
    fig.write_image(buf, format="png", scale=2)
    return base64.b64encode(buf.getvalue()).decode()


def build_html_report(df, profile, summary, charts, target_col=None):
    charts_html = ""
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

    return f"""<!DOCTYPE html>
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
</style></head><body>
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
<h2>Visualizations</h2>{charts_html}
</body></html>"""


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("""
    <div style="background:#378ADD;padding:20px 20px 16px;margin:-1rem -1rem 1.5rem">
      <div style="font-size:18px;font-weight:600;color:white">📊 Auto EDA</div>
      <div style="font-size:12px;color:rgba(255,255,255,0.75);margin-top:2px">Smart dataset explorer</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("**① Upload your dataset**")
    uploaded = st.file_uploader("CSV file", type=["csv"], label_visibility="collapsed")

    st.divider()

    st.markdown("**② AI provider**")
    provider = st.selectbox("Provider", list(PROVIDERS.keys()),
                            index=list(PROVIDERS.keys()).index(st.session_state.provider),
                            label_visibility="collapsed")
    st.session_state.provider = provider

    model = st.selectbox("Model", PROVIDERS[provider]["models"],
                         label_visibility="collapsed")
    st.session_state.model = model

    api_key = st.text_input("API Key", type="password",
                            placeholder=PROVIDERS[provider]["key_hint"],
                            value=st.session_state.api_key,
                            label_visibility="collapsed")
    st.session_state.api_key = api_key

    key_links = {
        "Gemini":             "https://aistudio.google.com",
        "OpenAI / ChatGPT":   "https://platform.openai.com/api-keys",
        "Claude (Anthropic)": "https://console.anthropic.com",
        "Grok (xAI)":         "https://console.x.ai",
    }
    if api_key:
        st.success(f"✓ {provider} key saved")
    else:
        st.caption(f"[Get free key →]({key_links[provider]})")

    st.divider()

    target_col = None
    do_clean   = False

    if uploaded:
        st.markdown("**③ Configure analysis**")
        df_peek     = pd.read_csv(uploaded, nrows=5)
        uploaded.seek(0)
        col_options = ["— none —"] + list(df_peek.columns)
        target_sel  = st.selectbox("🎯 Target variable", col_options, label_visibility="collapsed")
        target_col  = None if target_sel == "— none —" else target_sel

        do_clean = st.toggle("🧹 Auto-clean dataset", value=False)
        if do_clean:
            st.caption("Drops IDs & high-missing cols, imputes nulls, encodes binary, extracts dates, log-transforms skewed.")

    st.divider()
    st.markdown("""
    <div style="font-size:11px;color:#aaa;line-height:2">
      🔵 numeric &nbsp; 🟢 categorical<br>
      🟣 datetime &nbsp; 🟡 binary &nbsp; ⚫ id
    </div>
    """, unsafe_allow_html=True)


# ── Landing screen ────────────────────────────────────────────────────────────

if not uploaded:
    st.markdown("""
    <div style="max-width:560px;margin:60px auto 0;text-align:center">
      <div style="font-size:48px;margin-bottom:20px">📊</div>
      <div style="font-size:26px;font-weight:600;margin-bottom:10px;color:#1a1a1a">Auto EDA Tool</div>
      <div style="font-size:15px;color:#666;margin-bottom:36px;line-height:1.6">
        Upload any CSV and get instant analysis — smart type detection,
        interactive charts, target variable analysis, auto-clean,
        and an AI-written report.
      </div>
      <div style="display:grid;grid-template-columns:1fr 1fr;gap:12px;text-align:left;margin-bottom:36px">
        <div style="background:#fff;border:1px solid #e8ecf0;border-radius:12px;padding:16px">
          <div style="font-size:20px;margin-bottom:6px">🔍</div>
          <div style="font-size:13px;font-weight:500;margin-bottom:4px">Smart profiling</div>
          <div style="font-size:12px;color:#888">Auto-detects numeric, categorical, datetime, binary & ID columns</div>
        </div>
        <div style="background:#fff;border:1px solid #e8ecf0;border-radius:12px;padding:16px">
          <div style="font-size:20px;margin-bottom:6px">📈</div>
          <div style="font-size:13px;font-weight:500;margin-bottom:4px">Rich charts</div>
          <div style="font-size:12px;color:#888">Histograms, boxplots, bar charts, correlation matrix & missing map</div>
        </div>
        <div style="background:#fff;border:1px solid #e8ecf0;border-radius:12px;padding:16px">
          <div style="font-size:20px;margin-bottom:6px">🎯</div>
          <div style="font-size:13px;font-weight:500;margin-bottom:4px">Target analysis</div>
          <div style="font-size:12px;color:#888">Feature correlations, class balance & task type detection</div>
        </div>
        <div style="background:#fff;border:1px solid #e8ecf0;border-radius:12px;padding:16px">
          <div style="font-size:20px;margin-bottom:6px">🧹</div>
          <div style="font-size:13px;font-weight:500;margin-bottom:4px">Auto-clean</div>
          <div style="font-size:12px;color:#888">One-click imputation, encoding & feature engineering</div>
        </div>
      </div>
      <div style="font-size:13px;color:#aaa">← Upload a CSV in the sidebar to begin</div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()


# ── Main ──────────────────────────────────────────────────────────────────────

with st.spinner("Profiling dataset..."):
    df, profile = profile_csv(uploaded)

fname      = uploaded.name
type_count = {}
for m in profile["columns"].values():
    type_count[m["type"]] = type_count.get(m["type"], 0) + 1
type_str = "  ·  ".join(f"{v} {k}" for k, v in type_count.items())

st.markdown(f"""
<div style="background:#fff;border:1px solid #e8ecf0;border-radius:14px;
     padding:14px 20px;margin-bottom:20px;display:flex;align-items:center;gap:16px">
  <div style="font-size:24px">📄</div>
  <div>
    <div style="font-size:14px;font-weight:500;color:#1a1a1a">{fname}</div>
    <div style="font-size:12px;color:#888;margin-top:2px">{type_str}</div>
  </div>
  {"<div style='margin-left:auto;background:#d4eaff;color:#185FA5;border-radius:20px;padding:4px 14px;font-size:12px;font-weight:500'>🎯 Target: " + target_col + "</div>" if target_col else ""}
  {"<div style='margin-left:" + ("8px" if target_col else "auto") + ";background:#d4f5e9;color:#0a5c3e;border-radius:20px;padding:4px 14px;font-size:12px;font-weight:500'>🧹 Auto-clean on</div>" if do_clean else ""}
</div>
""", unsafe_allow_html=True)

m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Rows",       f"{profile['shape'][0]:,}")
m2.metric("Columns",    profile["shape"][1])
m3.metric("Missing",    f"{sum(profile['missing'].values()):,}")
m4.metric("Duplicates", profile["duplicates"])
m5.metric("Size",       f"{profile.get('memory_mb','?')} MB")

st.markdown("<div style='margin-bottom:8px'></div>", unsafe_allow_html=True)

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🗂 Columns", "📈 Charts", "🔬 Explore", "🎯 Target Analysis",
    "🧹 Auto-Clean", "🤖 AI Summary + Report"
])


# ── Tab 1: Columns ────────────────────────────────────────────────────────────

with tab1:
    type_colors   = {"numeric": "🔵", "categorical": "🟢", "datetime": "🟣", "binary": "🟡", "id": "⚫"}
    total_missing = sum(profile["missing"].values())
    if total_missing > 0:
        st.markdown(f"""
        <div style="background:#fff;border:1px solid #e8ecf0;border-radius:12px;
             padding:12px 16px;margin-bottom:16px;border-left:4px solid #E8963A">
          ⚠️ <strong>{total_missing:,} missing values</strong> detected across
          {sum(1 for m in profile['missing'].values() if m > 0)} columns
        </div>""", unsafe_allow_html=True)

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
                    st.warning(f"Skewness: {meta['skewness']} ({meta['skew_label']}) — {meta['outliers']} outliers detected")
            elif t == "datetime":
                st.write(f"**Range:** {meta['min']}  →  {meta['max']}")
                st.write(f"**Span:** {meta['range_days']} days")
            elif t == "binary":
                for val, pct in meta["values"].items():
                    st.write(f"`{val}` — {round(pct*100,1)}%")
            elif t == "id":
                st.info(f"Likely an ID column ({meta['unique']} unique values, {meta['unique_pct']}% of rows). Will be dropped in auto-clean.")
            else:
                st.write(f"**Unique values:** {meta['unique']} ({meta['unique_pct']}% of rows)")
                st.dataframe(pd.DataFrame.from_dict(meta["top5"], orient="index", columns=["count"]), use_container_width=True)
            if meta["nulls"] > 0:
                st.warning(f"{meta['nulls']} missing values ({meta['null_pct']}% of rows)")


# ── Tab 2: Charts ─────────────────────────────────────────────────────────────

with tab2:
    with st.spinner("Generating charts..."):
        charts = generate_charts(df, profile)

    sections = {
        "numeric":     "Numeric distributions",
        "boxplot":     "Boxplots",
        "categorical": "Categorical breakdowns",
        "binary":      "Binary splits",
        "datetime":    "Time series",
        "missing":     "Missing value map",
        "corr":        "Correlation matrix",
    }
    for type_key, section_title in sections.items():
        section_charts = [(k, c, f) for k, c, f in charts if k == type_key]
        if not section_charts:
            continue
        st.subheader(section_title)
        left, right = st.columns(2)
        for i, (_, col, fig) in enumerate(section_charts):
            (left if i % 2 == 0 else right).plotly_chart(fig, use_container_width=True)

# ── Tab 3: Explore ────────────────────────────────────────────────────────────

with tab3:
    st.markdown("#### Custom chart explorer")
    st.caption("Pick any two columns and chart type — find your own insights.")

    all_cols = list(profile["columns"].keys())
    num_cols = [c for c, m in profile["columns"].items() if m["type"] == "numeric"]
    cat_cols = [c for c, m in profile["columns"].items() if m["type"] in ("categorical", "binary")]

    c1, c2, c3 = st.columns(3)

    with c1:
        chart_type = st.selectbox("Chart type", [
            "Scatter plot",
            "Box plot (cat vs num)",
            "Bar chart (mean)",
            "Histogram",
            "Line chart",
            "Violin plot",
            "Heatmap (2 categoricals)",
        ])

    with c2:
        x_col = st.selectbox("X axis", all_cols, index=0)

    with c3:
        remaining = [c for c in all_cols if c != x_col]
        y_col = st.selectbox("Y axis", ["— none —"] + remaining, index=0)
        y_col = None if y_col == "— none —" else y_col

    # Optional color grouping
    color_col = st.selectbox(
        "Color / group by (optional)",
        ["— none —"] + cat_cols,
        index=0,
    )
    color_col = None if color_col == "— none —" else color_col

    st.divider()

    try:
        fig = None

        if chart_type == "Scatter plot":
            if not y_col:
                st.warning("Pick a Y axis for scatter plot.")
            else:
                fig = px.scatter(
                    df, x=x_col, y=y_col, color=color_col,
                    opacity=0.6, trendline="ols" if x_col in num_cols and y_col in num_cols else None,
                    title=f"{x_col} vs {y_col}",
                    color_discrete_sequence=px.colors.qualitative.Set2,
                )
                # Insight
                if x_col in num_cols and y_col in num_cols:
                    corr = round(float(df[[x_col, y_col]].corr().iloc[0, 1]), 3)
                    strength = "strong" if abs(corr) > 0.7 else "moderate" if abs(corr) > 0.4 else "weak"
                    direction = "positive" if corr > 0 else "negative"
                    st.markdown(f"""
                    <div style="background:#f0f7ff;border:1px solid #c8dff7;border-radius:8px;
                         padding:10px 14px;margin-bottom:12px;font-size:13px">
                      📊 Correlation: <strong>{corr}</strong> — {strength} {direction} relationship
                    </div>""", unsafe_allow_html=True)

        elif chart_type == "Box plot (cat vs num)":
            if not y_col:
                st.warning("Pick a Y axis (numeric) for box plot.")
            else:
                fig = px.box(
                    df, x=x_col, y=y_col, color=color_col,
                    title=f"{y_col} distribution by {x_col}",
                    color_discrete_sequence=px.colors.qualitative.Set2,
                )
                # Insight
                if y_col in num_cols:
                    grp_means = df.groupby(x_col)[y_col].mean().sort_values(ascending=False)
                    best  = grp_means.index[0]
                    worst = grp_means.index[-1]
                    st.markdown(f"""
                    <div style="background:#f0f7ff;border:1px solid #c8dff7;border-radius:8px;
                         padding:10px 14px;margin-bottom:12px;font-size:13px">
                      📊 Highest mean <code>{y_col}</code>: <strong>{best}</strong> ({round(grp_means[best],2)})
                      &nbsp;·&nbsp; Lowest: <strong>{worst}</strong> ({round(grp_means[worst],2)})
                    </div>""", unsafe_allow_html=True)

        elif chart_type == "Bar chart (mean)":
            if not y_col:
                st.warning("Pick a Y axis (numeric) for bar chart.")
            else:
                grp = df.groupby(x_col)[y_col].mean().reset_index().sort_values(y_col, ascending=False).head(20)
                fig = px.bar(
                    grp, x=x_col, y=y_col,
                    title=f"Mean {y_col} by {x_col}",
                    color_discrete_sequence=["#378ADD"],
                )
                st.markdown(f"""
                <div style="background:#f0f7ff;border:1px solid #c8dff7;border-radius:8px;
                     padding:10px 14px;margin-bottom:12px;font-size:13px">
                  📊 Showing mean <code>{y_col}</code> per <code>{x_col}</code> — top 20 categories
                </div>""", unsafe_allow_html=True)

        elif chart_type == "Histogram":
            fig = px.histogram(
                df, x=x_col, color=color_col,
                nbins=30, barmode="overlay",
                title=f"Distribution of {x_col}",
                color_discrete_sequence=px.colors.qualitative.Set2,
                opacity=0.75,
            )
            if x_col in num_cols:
                skew = round(float(df[x_col].skew()), 3)
                skew_label = "right-skewed — consider log transform" if skew > 1 else "left-skewed" if skew < -1 else "approximately normal"
                st.markdown(f"""
                <div style="background:#f0f7ff;border:1px solid #c8dff7;border-radius:8px;
                     padding:10px 14px;margin-bottom:12px;font-size:13px">
                  📊 Skewness: <strong>{skew}</strong> — {skew_label}
                </div>""", unsafe_allow_html=True)

        elif chart_type == "Line chart":
            if not y_col:
                st.warning("Pick a Y axis for line chart.")
            else:
                line_df = df[[x_col, y_col] + ([color_col] if color_col else [])].dropna().sort_values(x_col)
                fig = px.line(
                    line_df, x=x_col, y=y_col, color=color_col,
                    title=f"{y_col} over {x_col}",
                    color_discrete_sequence=px.colors.qualitative.Set2,
                )

        elif chart_type == "Violin plot":
            if not y_col:
                st.warning("Pick a Y axis (numeric) for violin plot.")
            else:
                fig = px.violin(
                    df, x=x_col, y=y_col, color=color_col,
                    box=True, points="outliers",
                    title=f"{y_col} by {x_col}",
                    color_discrete_sequence=px.colors.qualitative.Set2,
                )

        elif chart_type == "Heatmap (2 categoricals)":
            if not y_col:
                st.warning("Pick a Y axis for heatmap.")
            else:
                pivot = df.groupby([x_col, y_col]).size().reset_index(name="count")
                pivot = pivot.pivot(index=y_col, columns=x_col, values="count").fillna(0)
                fig = px.imshow(
                    pivot, text_auto=True,
                    color_continuous_scale="Blues",
                    title=f"Count heatmap: {x_col} × {y_col}",
                    aspect="auto",
                )

        if fig:
            layout = clean_layout(height=420)
            layout["title"] = {"text": fig.layout.title.text, "font": {"size": 14}, "x": 0}
            fig.update_layout(**layout)
            st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"Chart error: {e} — try a different column combination.")

    # ── Ask AI about this chart ───────────────────────────────────────────
    if y_col and st.session_state.get("api_key"):
        st.divider()
        if st.button("🤖 Ask AI to explain this chart", key="explain_chart"):
            with st.spinner("Analysing..."):
                try:
                    explain_prompt = f"""You are a data analyst. A user is exploring a dataset and plotted:
- Chart type: {chart_type}
- X axis: {x_col} ({profile['columns'].get(x_col, {}).get('type', 'unknown')})
- Y axis: {y_col} ({profile['columns'].get(y_col, {}).get('type', 'unknown')})
- Color group: {color_col}
- Dataset: {profile['shape'][0]:,} rows

Sample data for these columns:
{df[[c for c in [x_col, y_col, color_col] if c]].head(20).to_string()}

In 3-5 bullet points, explain:
- What pattern or insight this chart reveals
- Whether the relationship is expected or surprising
- What business action or next analysis step this suggests
Be specific. Reference actual values."""

                    insight = call_llm(explain_prompt, st.session_state.provider,
                                       st.session_state.api_key, st.session_state.model)
                    st.markdown(f"""
                    <div style="background:#f0f7ff;border:1px solid #c8dff7;border-radius:10px;
                         padding:16px 20px;font-size:13px;margin-top:8px">
                    {insight}
                    </div>""", unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"Error: {e}")

# ── Tab 4: Target Analysis ────────────────────────────────────────────────────

with tab4:
    if not target_col:
        st.markdown("""
        <div style="text-align:center;padding:48px 0">
          <div style="font-size:36px;margin-bottom:12px">🎯</div>
          <div style="font-size:15px;font-weight:500;margin-bottom:6px">No target variable selected</div>
          <div style="font-size:13px;color:#888">Choose a target column in the sidebar to see feature correlations,
          class balance, and task type detection.</div>
        </div>""", unsafe_allow_html=True)
    else:
        with st.spinner(f"Analysing target: {target_col}..."):
            task, t_charts, insights = analyze_target(df, profile, target_col)

        c1, c2 = st.columns([1, 3])
        c1.markdown(f"""
        <div style="background:#d4eaff;border-radius:12px;padding:16px;text-align:center">
          <div style="font-size:11px;color:#185FA5;font-weight:500;text-transform:uppercase;letter-spacing:.06em">Task type</div>
          <div style="font-size:20px;font-weight:600;color:#185FA5;margin-top:4px">{task.capitalize()}</div>
        </div>""", unsafe_allow_html=True)

        with c2:
            for ins in insights:
                st.markdown(f"""
                <div style="background:#fff;border:1px solid #e8ecf0;border-radius:10px;
                     padding:10px 16px;margin-bottom:8px;font-size:13px">{ins}</div>
                """, unsafe_allow_html=True)

        st.divider()
        left, right = st.columns(2)
        for i, (_, col, fig) in enumerate(t_charts):
            (left if i % 2 == 0 else right).plotly_chart(fig, use_container_width=True)


# ── Tab 5: Auto-Clean ─────────────────────────────────────────────────────────

with tab5:
    if not do_clean:
        st.markdown("""
        <div style="text-align:center;padding:48px 0">
          <div style="font-size:36px;margin-bottom:12px">🧹</div>
          <div style="font-size:15px;font-weight:500;margin-bottom:6px">Auto-clean is off</div>
          <div style="font-size:13px;color:#888">Toggle <strong>Clean & prepare dataset</strong> in the sidebar
          to automatically impute, encode, and transform your data.</div>
        </div>""", unsafe_allow_html=True)
    else:
        with st.spinner("Cleaning dataset..."):
            cleaned_df, clean_log = auto_clean(df, profile, target_col)

        ca, cb, cc = st.columns(3)
        ca.metric("Columns before", df.shape[1])
        cb.metric("Columns after",  cleaned_df.shape[1])
        cc.metric("Rows after",     f"{len(cleaned_df):,}")

        st.markdown("<div style='margin:16px 0 8px;font-size:13px;font-weight:500'>What was done:</div>", unsafe_allow_html=True)
        for entry in clean_log:
            st.markdown(f"""
            <div style="background:#fff;border:1px solid #e8ecf0;border-radius:8px;
                 padding:8px 14px;margin-bottom:6px;font-size:13px">{entry}</div>
            """, unsafe_allow_html=True)

        st.divider()
        st.subheader("Preview — first 20 rows")
        st.dataframe(cleaned_df.head(20), use_container_width=True)
        st.download_button(
            label="⬇️  Download cleaned CSV",
            data=cleaned_df.to_csv(index=False).encode("utf-8"),
            file_name="cleaned_data.csv",
            mime="text/csv",
        )


# ── Tab 6: AI Summary + Report ────────────────────────────────────────────────

with tab6:
    if not st.session_state.api_key:
        st.markdown(f"""
        <div style="text-align:center;padding:48px 0">
          <div style="font-size:36px;margin-bottom:12px">🤖</div>
          <div style="font-size:15px;font-weight:500;margin-bottom:6px">No API key yet</div>
          <div style="font-size:13px;color:#888;margin-bottom:16px">
            Add your {st.session_state.provider} API key in the sidebar to generate the AI analysis.
          </div>
        </div>""", unsafe_allow_html=True)
    else:
        prov  = st.session_state.provider
        mdl   = st.session_state.model
        a_key = st.session_state.api_key

        st.markdown(f"""
        <div style="background:#fff;border:1px solid #e8ecf0;border-radius:10px;
             padding:10px 16px;margin-bottom:16px;font-size:13px;color:#555">
          Using <strong>{prov}</strong> / <code>{mdl}</code>
          {"&nbsp;·&nbsp; 🎯 Target: <strong>" + target_col + "</strong>" if target_col else ""}
        </div>""", unsafe_allow_html=True)

        with st.spinner(f"Generating analysis with {prov}..."):
            try:
                summary = generate_summary(profile, target_col, prov, a_key, mdl, _df=df)
                st.markdown(summary)
                st.divider()
                with st.spinner("Building downloadable report..."):
                    html = build_html_report(df, profile, summary, charts, target_col)
                st.download_button(
                    label="⬇️  Download full report (.html)",
                    data=html,
                    file_name="eda_report.html",
                    mime="text/html",
                )
            except Exception as e:
                st.error(f"Error from {prov}: {e}")
