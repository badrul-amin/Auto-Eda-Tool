import json
import base64
import io
from datetime import datetime

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats

st.set_page_config(page_title="Auto EDA", layout="wide", initial_sidebar_state="expanded")


# ── CSS ───────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
  .stApp { background: #f4f6f9; }
  .block-container { padding-top: 1.5rem !important; }
  [data-testid="stSidebar"] { background: #ffffff; border-right: 1px solid #e8ecf0; }
  [data-testid="stMetric"] { background: #ffffff; border: 1px solid #e8ecf0; border-radius: 12px; padding: 16px 18px !important; }
  [data-testid="stMetricLabel"] { font-size: 12px !important; color: #888 !important; }
  [data-testid="stMetricValue"] { font-size: 22px !important; font-weight: 600 !important; }
  .stTabs [data-baseweb="tab-list"] { background: #ffffff; border-radius: 12px; padding: 4px; border: 1px solid #e8ecf0; gap: 2px; margin-bottom: 16px; }
  .stTabs [data-baseweb="tab"] { font-size: 13px; font-weight: 500; color: #888; padding: 8px 20px; border-radius: 8px; }
  .stTabs [aria-selected="true"] { background: #378ADD !important; color: white !important; }
  .stButton > button { border-radius: 8px; font-size: 13px; font-weight: 500; padding: 8px 20px; background: #378ADD; color: white !important; border: none; width: 100%; }
  .stButton > button:hover { background: #2567b8 !important; }
  .stDownloadButton > button { border-radius: 8px; font-size: 13px; font-weight: 500; padding: 10px 20px; background: #1D9E75; color: white !important; border: none; }
  .stDownloadButton > button:hover { background: #157a5a !important; }
  .streamlit-expanderHeader { background: #ffffff !important; border-radius: 10px !important; font-size: 13px; font-weight: 500; }
  #MainMenu, footer { visibility: hidden; }
  .insight-box { background: #f0f7ff; border: 1px solid #c8dff7; border-radius: 10px; padding: 12px 16px; font-size: 13px; margin-bottom: 8px; }
  .warn-box { background: #fff8f0; border: 1px solid #fcd9b0; border-radius: 10px; padding: 12px 16px; font-size: 13px; margin-bottom: 8px; }
  .stat-sig { color: #1D9E75; font-weight: 600; }
  .stat-not { color: #E05A3A; font-weight: 600; }
</style>
""", unsafe_allow_html=True)


# ── Constants ─────────────────────────────────────────────────────────────────

NA_VALUES = [
    "", " ", "NA", "N/A", "n/a", "na", "NULL", "null", "None", "none",
    "NaN", "nan", "-", "--", "?", "missing", "MISSING", "unknown",
    "Unknown", "UNK", "#N/A", "#NULL!",
]
ID_HINTS   = ["id", "uuid", "guid", "key", "ref", "index", "no", "num",
              "number", "code", "item", "sku", "barcode"]
DATE_HINTS = ["date", "time", "datetime", "timestamp", "created", "updated", "at", "on"]
CAT_HINTS  = ["type", "status", "category", "flag", "gender", "sex", "store",
              "region", "country", "zip", "postal", "grade", "class", "label",
              "group", "quarter", "week", "segment", "tier", "level", "rank", "phase"]

PROVIDERS = {
    "Gemini":             {"models": ["gemini-1.5-flash", "gemini-1.5-pro", "gemini-2.0-flash"], "key_hint": "AIza..."},
    "OpenAI / ChatGPT":   {"models": ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"],                 "key_hint": "sk-..."},
    "Claude (Anthropic)": {"models": ["claude-haiku-4-5-20251001", "claude-sonnet-4-6"],          "key_hint": "sk-ant-..."},
    "Grok (xAI)":         {"models": ["grok-3-mini", "grok-3"],                                   "key_hint": "xai-..."},
}


# ── Session state ─────────────────────────────────────────────────────────────

for key, val in {
    "provider": "Gemini", "model": "gemini-1.5-flash",
    "api_key": "", "chat_history": [], "working_df": None,
    "is_aggregated": False,
}.items():
    if key not in st.session_state:
        st.session_state[key] = val


# ── LLM caller ────────────────────────────────────────────────────────────────

def call_llm(prompt, provider, api_key, model, system=None):
    if provider == "Gemini":
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        full = f"{system}\n\n{prompt}" if system else prompt
        return genai.GenerativeModel(model).generate_content(full).text

    elif provider == "OpenAI / ChatGPT":
        from openai import OpenAI
        msgs = []
        if system:
            msgs.append({"role": "system", "content": system})
        msgs.append({"role": "user", "content": prompt})
        r = OpenAI(api_key=api_key).chat.completions.create(
            model=model, messages=msgs, max_tokens=1500)
        return r.choices[0].message.content

    elif provider == "Claude (Anthropic)":
        import anthropic
        kwargs = {"model": model, "max_tokens": 1500,
                  "messages": [{"role": "user", "content": prompt}]}
        if system:
            kwargs["system"] = system
        return anthropic.Anthropic(api_key=api_key).messages.create(**kwargs).content[0].text

    elif provider == "Grok (xAI)":
        from openai import OpenAI
        msgs = []
        if system:
            msgs.append({"role": "system", "content": system})
        msgs.append({"role": "user", "content": prompt})
        r = OpenAI(api_key=api_key, base_url="https://api.x.ai/v1").chat.completions.create(
            model=model, messages=msgs, max_tokens=1500)
        return r.choices[0].message.content

    raise ValueError(f"Unknown provider: {provider}")


# ── Profiler ──────────────────────────────────────────────────────────────────

@st.cache_data
def profile_df(df_json):
    df = pd.read_json(io.StringIO(df_json), orient="split")
    n_rows = len(df)
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
            h == col_lower or f"_{h}" in col_lower or
            f"{h}_" in col_lower or h in col_lower.split("_")
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
            n_out      = int(((series < q1 - 1.5*iqr) | (series > q3 + 1.5*iqr)).sum())
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
    return profile


# ── Chart layout ──────────────────────────────────────────────────────────────

def clean_layout(height=300, title=""):
    return dict(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="system-ui, sans-serif", size=12, color="#555"),
        title=dict(text=title, font=dict(size=13), x=0),
        margin=dict(l=0, r=0, t=40, b=0),
        xaxis=dict(showgrid=True, gridcolor="#f0f0f0", zeroline=False),
        yaxis=dict(showgrid=False, zeroline=False),
        showlegend=False, height=height,
    )


# ── Auto charts ───────────────────────────────────────────────────────────────

@st.cache_data
def generate_charts(_df, profile):
    charts    = []
    num_cols  = [c for c, m in profile["columns"].items() if m["type"] == "numeric"]
    cat_cols  = [c for c, m in profile["columns"].items() if m["type"] == "categorical"]
    bin_cols  = [c for c, m in profile["columns"].items() if m["type"] == "binary"]
    date_cols = [c for c, m in profile["columns"].items() if m["type"] == "datetime"]

    for col in num_cols:
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=_df[col].dropna(), nbinsx=30,
                                   marker_color="#378ADD", opacity=0.85, name=col))
        fig.update_layout(**clean_layout(title=f"Distribution: {col}"))
        charts.append(("numeric", col, fig))

        fig2 = go.Figure()
        fig2.add_trace(go.Box(x=_df[col].dropna(), marker_color="#378ADD",
                              boxmean=True, name=col))
        fig2.update_layout(**clean_layout(height=200, title=f"Boxplot: {col}"))
        charts.append(("boxplot", col, fig2))

    for col in cat_cols:
        top = _df[col].value_counts().head(15).reset_index()
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
        fig.update_layout(**clean_layout(height=240, title=f"Split: {col}"))
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
                        title="Missing value map", aspect="auto")
        fig.update_layout(**clean_layout(height=max(200, len(missing_cols)*20+60)))
        fig.update_coloraxes(showscale=False)
        charts.append(("missing", "missing_map", fig))

    if len(num_cols) > 1:
        corr = _df[num_cols].corr().round(2)
        fig  = px.imshow(corr, text_auto=True, color_continuous_scale="Blues",
                         title="Correlation matrix", aspect="auto")
        fig.update_layout(**clean_layout(height=max(300, len(num_cols)*40+80)))
        charts.append(("corr", "correlation", fig))

    return charts


# ── Time series ───────────────────────────────────────────────────────────────

def render_timeseries(df, profile):
    date_cols = [c for c, m in profile["columns"].items() if m["type"] == "datetime"]
    num_cols  = [c for c, m in profile["columns"].items() if m["type"] == "numeric"]

    if not date_cols:
        # Try to find date-like object columns
        for col in df.select_dtypes(include="object").columns:
            try:
                parsed = pd.to_datetime(df[col], errors="coerce")
                if parsed.notna().mean() > 0.7:
                    df[col] = parsed
                    date_cols.append(col)
            except Exception:
                pass

    if not date_cols or not num_cols:
        st.info("No datetime + numeric columns detected for time series analysis.")
        return

    st.markdown("#### 📅 Time series analysis")

    c1, c2, c3 = st.columns(3)
    date_col   = c1.selectbox("Date column",   date_cols, key="ts_date")
    value_col  = c2.selectbox("Value column",  num_cols,  key="ts_val")
    freq_label = c3.selectbox("Frequency", ["Auto", "Daily", "Weekly", "Monthly", "Quarterly"], key="ts_freq")

    cat_cols = [c for c, m in profile["columns"].items() if m["type"] == "categorical"]
    group_col = st.selectbox("Group by (optional)", ["— none —"] + cat_cols, key="ts_group")
    group_col = None if group_col == "— none —" else group_col

    try:
        ts_df = df[[date_col, value_col] + ([group_col] if group_col else [])].copy()
        ts_df[date_col] = pd.to_datetime(ts_df[date_col], errors="coerce")
        ts_df = ts_df.dropna(subset=[date_col])

        span = (ts_df[date_col].max() - ts_df[date_col].min()).days
        freq_map = {"Auto": "D" if span <= 90 else "W" if span <= 365 else "ME",
                    "Daily": "D", "Weekly": "W", "Monthly": "ME", "Quarterly": "QE"}
        freq = freq_map[freq_label]

        if group_col:
            groups  = ts_df[group_col].dropna().unique()
            fig     = go.Figure()
            all_agg = []
            for g in groups[:8]:
                sub = ts_df[ts_df[group_col] == g].set_index(date_col)[value_col]
                agg = sub.resample(freq).sum().reset_index()
                agg.columns = [date_col, value_col]
                agg[group_col] = g
                all_agg.append(agg)
                fig.add_trace(go.Scatter(x=agg[date_col], y=agg[value_col],
                                         mode="lines+markers", name=str(g)))
            fig.update_layout(**clean_layout(height=380, title=f"{value_col} over time by {group_col}"))
            st.plotly_chart(fig, use_container_width=True)
            combined = pd.concat(all_agg)
        else:
            agg = ts_df.set_index(date_col)[value_col].resample(freq).sum().reset_index()
            agg.columns = [date_col, value_col]

            # Moving average
            window = max(3, len(agg) // 10)
            agg["MA"] = agg[value_col].rolling(window=window, center=True).mean()

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=agg[date_col], y=agg[value_col],
                                     mode="lines", name=value_col,
                                     line=dict(color="#378ADD", width=1.5), opacity=0.6))
            fig.add_trace(go.Scatter(x=agg[date_col], y=agg["MA"],
                                     mode="lines", name=f"Moving avg ({window})",
                                     line=dict(color="#E8963A", width=2.5)))
            fig.update_layout(**clean_layout(height=350,
                              title=f"{value_col} over time (with {window}-period moving average)"))
            fig.update_layout(showlegend=True)
            st.plotly_chart(fig, use_container_width=True)

            # Seasonality
            col1, col2 = st.columns(2)
            try:
                ts_df["month"] = ts_df[date_col].dt.month_name()
                monthly = ts_df.groupby("month")[value_col].mean().reindex(
                    ["January","February","March","April","May","June",
                     "July","August","September","October","November","December"]
                ).dropna().reset_index()
                fig2 = px.bar(monthly, x="month", y=value_col,
                              color_discrete_sequence=["#7F77DD"],
                              title=f"Average {value_col} by month")
                fig2.update_layout(**clean_layout(height=280))
                col1.plotly_chart(fig2, use_container_width=True)
            except Exception:
                pass

            try:
                ts_df["dow"] = ts_df[date_col].dt.day_name()
                dow_order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
                daily = ts_df.groupby("dow")[value_col].mean().reindex(dow_order).dropna().reset_index()
                fig3 = px.bar(daily, x="dow", y=value_col,
                              color_discrete_sequence=["#1D9E75"],
                              title=f"Average {value_col} by day of week")
                fig3.update_layout(**clean_layout(height=280))
                col2.plotly_chart(fig3, use_container_width=True)
            except Exception:
                pass

            # Anomaly detection — simple z-score
            agg["zscore"] = np.abs((agg[value_col] - agg[value_col].mean()) / agg[value_col].std())
            anomalies = agg[agg["zscore"] > 2.5]
            if not anomalies.empty:
                st.markdown(f"""
                <div class="warn-box">
                  ⚠️ <strong>{len(anomalies)} anomalies detected</strong> (z-score > 2.5) —
                  periods with unusually high or low {value_col}:<br>
                  {', '.join(str(d)[:10] for d in anomalies[date_col].tolist()[:5])}
                </div>""", unsafe_allow_html=True)

            # YoY / MoM growth
            if len(agg) >= 2:
                latest  = agg[value_col].iloc[-1]
                prev    = agg[value_col].iloc[-2]
                growth  = round((latest - prev) / max(abs(prev), 1) * 100, 1)
                arrow   = "↑" if growth > 0 else "↓"
                color   = "#1D9E75" if growth > 0 else "#E05A3A"
                period  = "MoM" if freq in ("ME","M") else "WoW" if freq == "W" else "DoD"
                st.markdown(f"""
                <div class="insight-box">
                  📈 Latest {period} change: <strong style="color:{color}">{arrow} {abs(growth)}%</strong>
                  &nbsp;·&nbsp; Latest period value: <strong>{latest:,.2f}</strong>
                </div>""", unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Time series error: {e}")


# ── Statistical tests ─────────────────────────────────────────────────────────

def render_stat_tests(df, profile):
    st.markdown("#### 🧪 Statistical tests")
    st.caption("Test whether differences and relationships in your data are statistically significant — not just visual.")

    num_cols = [c for c, m in profile["columns"].items() if m["type"] == "numeric"]
    cat_cols = [c for c, m in profile["columns"].items() if m["type"] in ("categorical", "binary")]

    test_type = st.selectbox("Test type", [
        "Correlation significance (numeric vs numeric)",
        "Group difference — t-test (2 groups)",
        "Group difference — ANOVA (3+ groups)",
        "Association — Chi-square (categorical vs categorical)",
        "Run all auto-tests",
    ], key="stat_test_type")

    st.divider()

    def sig_badge(p):
        if p < 0.001:  return f'<span class="stat-sig">✅ p={p:.4f} — highly significant</span>'
        elif p < 0.05: return f'<span class="stat-sig">✅ p={p:.4f} — significant</span>'
        else:          return f'<span class="stat-not">❌ p={p:.4f} — not significant</span>'

    def plain_english(test, col_a, col_b, stat, p):
        sig = p < 0.05
        if test == "correlation":
            direction = "positively" if stat > 0 else "negatively"
            strength  = "strongly" if abs(stat) > 0.7 else "moderately" if abs(stat) > 0.4 else "weakly"
            return (f"`{col_a}` and `{col_b}` are {strength} {direction} correlated (r={stat:.3f}). "
                    f"{'This relationship is statistically significant.' if sig else 'However, this is NOT statistically significant — could be random.'}")
        elif test in ("ttest", "anova"):
            return (f"The mean `{col_b}` differs {'significantly' if sig else 'but NOT significantly'} "
                    f"across groups of `{col_a}`. "
                    f"{'Real difference exists.' if sig else 'Treat with caution — may be noise.'}")
        elif test == "chi2":
            return (f"`{col_a}` and `{col_b}` are {'associated' if sig else 'NOT associated'} — "
                    f"knowing one {'does' if sig else 'does NOT'} help predict the other.")
        return ""

    if test_type == "Correlation significance (numeric vs numeric)":
        if len(num_cols) < 2:
            st.info("Need at least 2 numeric columns.")
            return
        c1, c2 = st.columns(2)
        col_a = c1.selectbox("Column A", num_cols, key="corr_a")
        col_b = c2.selectbox("Column B", [c for c in num_cols if c != col_a], key="corr_b")
        paired = df[[col_a, col_b]].dropna()
        r, p   = stats.pearsonr(paired[col_a], paired[col_b])
        st.markdown(f"""
        <div class="insight-box">
          <strong>Pearson correlation: {r:.4f}</strong><br>
          {sig_badge(p)}<br>
          <span style="color:#666;font-size:12px">{plain_english("correlation", col_a, col_b, r, p)}</span>
        </div>""", unsafe_allow_html=True)
        fig = px.scatter(paired, x=col_a, y=col_b, trendline="ols",
                         color_discrete_sequence=["#378ADD"],
                         title=f"{col_a} vs {col_b} (r={r:.3f}, p={p:.4f})")
        fig.update_layout(**clean_layout(height=360))
        st.plotly_chart(fig, use_container_width=True)

    elif test_type == "Group difference — t-test (2 groups)":
        if not cat_cols or not num_cols:
            st.info("Need at least one categorical and one numeric column.")
            return
        c1, c2 = st.columns(2)
        cat_col = c1.selectbox("Group column (categorical)", cat_cols, key="tt_cat")
        num_col = c2.selectbox("Value column (numeric)",     num_cols, key="tt_num")
        groups  = df[cat_col].dropna().unique()
        if len(groups) < 2:
            st.warning("Need at least 2 groups.")
            return
        g1 = st.selectbox("Group 1", groups, index=0, key="tt_g1")
        g2 = st.selectbox("Group 2", [g for g in groups if g != g1], index=0, key="tt_g2")
        a  = df[df[cat_col] == g1][num_col].dropna()
        b  = df[df[cat_col] == g2][num_col].dropna()
        t, p = stats.ttest_ind(a, b)
        st.markdown(f"""
        <div class="insight-box">
          <strong>t-test: {g1} vs {g2} on {num_col}</strong><br>
          Mean {g1}: <strong>{a.mean():.3f}</strong> &nbsp;·&nbsp; Mean {g2}: <strong>{b.mean():.3f}</strong><br>
          Difference: <strong>{(a.mean()-b.mean()):.3f}</strong><br>
          {sig_badge(p)}<br>
          <span style="color:#666;font-size:12px">{plain_english("ttest", cat_col, num_col, t, p)}</span>
        </div>""", unsafe_allow_html=True)
        fig = px.box(df[df[cat_col].isin([g1, g2])], x=cat_col, y=num_col,
                     color=cat_col, color_discrete_sequence=["#378ADD", "#1D9E75"],
                     title=f"{num_col} — {g1} vs {g2}")
        fig.update_layout(**clean_layout(height=340))
        st.plotly_chart(fig, use_container_width=True)

    elif test_type == "Group difference — ANOVA (3+ groups)":
        if not cat_cols or not num_cols:
            st.info("Need at least one categorical and one numeric column.")
            return
        c1, c2  = st.columns(2)
        cat_col = c1.selectbox("Group column", cat_cols, key="anova_cat")
        num_col = c2.selectbox("Value column", num_cols, key="anova_num")
        groups  = [df[df[cat_col] == g][num_col].dropna()
                   for g in df[cat_col].dropna().unique()]
        groups  = [g for g in groups if len(g) > 1]
        if len(groups) < 2:
            st.warning("Need at least 2 groups with data.")
            return
        f, p = stats.f_oneway(*groups)
        st.markdown(f"""
        <div class="insight-box">
          <strong>One-way ANOVA: {num_col} across {cat_col}</strong><br>
          Groups tested: <strong>{len(groups)}</strong> &nbsp;·&nbsp; F-statistic: <strong>{f:.4f}</strong><br>
          {sig_badge(p)}<br>
          <span style="color:#666;font-size:12px">{plain_english("anova", cat_col, num_col, f, p)}</span>
        </div>""", unsafe_allow_html=True)
        fig = px.box(df, x=cat_col, y=num_col, color=cat_col,
                     color_discrete_sequence=px.colors.qualitative.Set2,
                     title=f"{num_col} by {cat_col} (ANOVA p={p:.4f})")
        fig.update_layout(**clean_layout(height=360))
        st.plotly_chart(fig, use_container_width=True)

    elif test_type == "Association — Chi-square (categorical vs categorical)":
        if len(cat_cols) < 2:
            st.info("Need at least 2 categorical columns.")
            return
        c1, c2  = st.columns(2)
        col_a   = c1.selectbox("Column A", cat_cols, key="chi_a")
        col_b   = c2.selectbox("Column B", [c for c in cat_cols if c != col_a], key="chi_b")
        ct      = pd.crosstab(df[col_a], df[col_b])
        chi2, p, dof, _ = stats.chi2_contingency(ct)
        st.markdown(f"""
        <div class="insight-box">
          <strong>Chi-square test: {col_a} vs {col_b}</strong><br>
          Chi² = <strong>{chi2:.4f}</strong> &nbsp;·&nbsp; Degrees of freedom: <strong>{dof}</strong><br>
          {sig_badge(p)}<br>
          <span style="color:#666;font-size:12px">{plain_english("chi2", col_a, col_b, chi2, p)}</span>
        </div>""", unsafe_allow_html=True)
        fig = px.imshow(ct, text_auto=True, color_continuous_scale="Blues",
                        title=f"Crosstab: {col_a} vs {col_b}", aspect="auto")
        fig.update_layout(**clean_layout(height=360))
        st.plotly_chart(fig, use_container_width=True)

    elif test_type == "Run all auto-tests":
        results = []

        # All numeric pairs correlation
        for i, ca in enumerate(num_cols):
            for cb in num_cols[i+1:]:
                try:
                    paired = df[[ca, cb]].dropna()
                    if len(paired) < 5:
                        continue
                    r, p = stats.pearsonr(paired[ca], paired[cb])
                    results.append({
                        "Test": "Pearson correlation",
                        "Column A": ca, "Column B": cb,
                        "Statistic": round(r, 4),
                        "p-value": round(p, 4),
                        "Significant": "✅ Yes" if p < 0.05 else "❌ No",
                        "Note": plain_english("correlation", ca, cb, r, p),
                    })
                except Exception:
                    pass

        # Cat vs num ANOVA
        for cat in cat_cols:
            for num in num_cols:
                try:
                    groups = [df[df[cat] == g][num].dropna()
                              for g in df[cat].dropna().unique()]
                    groups = [g for g in groups if len(g) > 1]
                    if len(groups) < 2:
                        continue
                    f, p = stats.f_oneway(*groups)
                    results.append({
                        "Test": "ANOVA",
                        "Column A": cat, "Column B": num,
                        "Statistic": round(f, 4),
                        "p-value": round(p, 4),
                        "Significant": "✅ Yes" if p < 0.05 else "❌ No",
                        "Note": plain_english("anova", cat, num, f, p),
                    })
                except Exception:
                    pass

        if results:
            result_df = pd.DataFrame(results).sort_values("p-value")
            sig_count = (result_df["Significant"] == "✅ Yes").sum()
            st.markdown(f"""
            <div class="insight-box">
              🧪 Ran <strong>{len(results)}</strong> tests —
              <strong>{sig_count}</strong> statistically significant (p &lt; 0.05)
            </div>""", unsafe_allow_html=True)
            st.dataframe(result_df, use_container_width=True, hide_index=True)
        else:
            st.info("Not enough data to run tests.")


# ── Target analysis ───────────────────────────────────────────────────────────

def render_target(df, profile, target_col):
    meta   = profile["columns"].get(target_col, {})
    t_type = meta.get("type", "categorical")
    task   = "classification" if t_type in ("binary","categorical") or \
             (t_type == "numeric" and df[target_col].nunique() <= 15) else "regression"

    c1, c2 = st.columns([1, 3])
    c1.markdown(f"""
    <div style="background:#d4eaff;border-radius:12px;padding:16px;text-align:center">
      <div style="font-size:11px;color:#185FA5;font-weight:500;text-transform:uppercase;letter-spacing:.06em">Task type</div>
      <div style="font-size:20px;font-weight:600;color:#185FA5;margin-top:4px">{task.capitalize()}</div>
    </div>""", unsafe_allow_html=True)

    insights = []
    charts   = []

    if task == "classification":
        vc = df[target_col].value_counts().reset_index()
        vc.columns = [target_col, "count"]
        vc["pct"] = (vc["count"] / len(df) * 100).round(1)
        fig = px.bar(vc, x=target_col, y="count",
                     text=vc["pct"].astype(str) + "%",
                     color_discrete_sequence=["#378ADD"],
                     title=f"Target distribution: {target_col}")
        fig.update_layout(**clean_layout(height=300))
        charts.append(fig)
        ratios = vc["count"] / vc["count"].sum()
        if ratios.min() < 0.1:
            insights.append(f"⚠️ **Class imbalance** — minority class is {round(ratios.min()*100,1)}%. Consider SMOTE.")
        else:
            insights.append("✅ Classes are reasonably balanced.")
    else:
        fig = px.histogram(df[target_col].dropna(), nbins=40,
                           color_discrete_sequence=["#7F77DD"],
                           title=f"Target distribution: {target_col}")
        fig.update_layout(**clean_layout(height=300))
        charts.append(fig)
        skew = round(float(df[target_col].skew()), 3)
        insights.append(f"{'⚠️ Target skewed (' + str(skew) + '). Consider log-transform.' if abs(skew) > 1 else '✅ Target looks normally distributed (skewness: ' + str(skew) + ').'}")

    num_cols = [c for c, m in profile["columns"].items() if m["type"] == "numeric" and c != target_col]
    cat_cols = [c for c, m in profile["columns"].items() if m["type"] in ("categorical","binary") and c != target_col]

    if num_cols:
        try:
            corr_vals = df[num_cols + [target_col]].corr()[target_col].drop(target_col).abs().sort_values(ascending=False)
            top_corr  = corr_vals.head(10).reset_index()
            top_corr.columns = ["feature", "correlation"]
            fig2 = px.bar(top_corr, x="correlation", y="feature", orientation="h",
                          color_discrete_sequence=["#1D9E75"],
                          title=f"Feature correlation with {target_col}")
            fig2.update_layout(**clean_layout(height=320))
            charts.append(fig2)
            if len(corr_vals) > 0:
                top_feat = corr_vals.index[0]
                insights.append(f"🔵 **Strongest predictor:** `{top_feat}` (r={round(corr_vals[top_feat],3)})")
        except Exception:
            pass

    with c2:
        for ins in insights:
            st.markdown(f'<div class="insight-box">{ins}</div>', unsafe_allow_html=True)

    left, right = st.columns(2)
    for i, fig in enumerate(charts):
        (left if i % 2 == 0 else right).plotly_chart(fig, use_container_width=True)

    for col in cat_cols[:3]:
        try:
            if task == "regression":
                grp = df.groupby(col)[target_col].mean().reset_index().sort_values(target_col, ascending=False).head(12)
                fig = px.bar(grp, x=target_col, y=col, orientation="h",
                             color_discrete_sequence=["#E8963A"],
                             title=f"Mean {target_col} by {col}")
            else:
                grp = df.groupby([col, target_col]).size().reset_index(name="count")
                fig = px.bar(grp, x=col, y="count", color=str(target_col),
                             barmode="group", title=f"{col} vs {target_col}",
                             color_discrete_sequence=px.colors.qualitative.Set2)
            fig.update_layout(**clean_layout(height=320))
            st.plotly_chart(fig, use_container_width=True)
        except Exception:
            pass


# ── Auto clean ────────────────────────────────────────────────────────────────

def auto_clean(df, profile, target_col=None):
    cleaned = df.copy()
    log     = []

    id_cols = [c for c, m in profile["columns"].items() if m["type"] == "id"]
    if id_cols:
        cleaned.drop(columns=id_cols, inplace=True, errors="ignore")
        log.append(f"🗑 Dropped ID columns: {', '.join(id_cols)}")

    high_miss = [c for c, m in profile["columns"].items()
                 if m["null_pct"] > 60 and c != target_col]
    if high_miss:
        cleaned.drop(columns=high_miss, inplace=True, errors="ignore")
        log.append(f"🗑 Dropped high-missing columns (>60%): {', '.join(high_miss)}")

    n_before = len(cleaned)
    cleaned.drop_duplicates(inplace=True)
    if len(cleaned) < n_before:
        log.append(f"🗑 Dropped {n_before - len(cleaned)} duplicate rows")

    for col in [c for c in cleaned.columns if c in profile["columns"]
                and profile["columns"][c]["type"] == "numeric"
                and cleaned[c].isnull().any()]:
        med = cleaned[col].median()
        cleaned[col].fillna(med, inplace=True)
        log.append(f"🔧 Imputed `{col}` with median ({round(med,3)})")

    for col in [c for c in cleaned.columns if c in profile["columns"]
                and profile["columns"][c]["type"] in ("categorical","binary")
                and cleaned[c].isnull().any()]:
        mode = cleaned[col].mode()[0]
        cleaned[col].fillna(mode, inplace=True)
        log.append(f"🔧 Imputed `{col}` with mode ('{mode}')")

    for col in [c for c in cleaned.columns if c in profile["columns"]
                and profile["columns"][c]["type"] == "numeric"
                and profile["columns"][c].get("skewness", 0) > 1
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


# ── AI Summary ────────────────────────────────────────────────────────────────

@st.cache_data
def generate_summary(profile, target_col, provider, api_key, model, sample_json, is_aggregated):
    sample     = pd.read_json(io.StringIO(sample_json), orient="split")
    sample_str = sample.head(15).to_string()
    stats_str  = sample.describe(include="all").round(2).to_string()
    target     = f"Target variable: `{target_col}`" if target_col else "No target variable selected."
    agg_note   = "Note: This data has been aggregated by the user before analysis." if is_aggregated else ""

    col_types = "\n".join(
        f"- `{col}`: {meta['type']} | {meta['nulls']} nulls ({meta['null_pct']}%)"
        + (f" | skewness={meta.get('skewness','N/A')} | outliers={meta.get('outliers','N/A')}" if meta["type"] == "numeric" else "")
        + (f" | top values: {list(meta.get('top5',{}).keys())}" if meta["type"] in ("categorical","id") else "")
        for col, meta in profile["columns"].items()
    )

    prompt = f"""You are a senior data scientist doing EDA on a real business dataset.

{target}
{agg_note}

Dataset: {profile['shape'][0]:,} rows × {profile['shape'][1]} columns | {profile.get('memory_mb','?')} MB

Column summary:
{col_types}

Actual data sample (first 15 rows):
{sample_str}

Descriptive statistics:
{stats_str}

Write a thorough, specific EDA report in markdown:

## Overview
What this dataset appears to be about. Shape, quality score (0-100), key observations.

## Data Quality
Every column with issues — missing values, wrong types, suspicious values. Exact recommended fix per column.

## Key Patterns & Business Insights
What do the actual values reveal? Reference real numbers. Spot patterns a business analyst would care about.

## Target Variable Analysis
{('Task type, class balance, strongest predictors, leakage risks.' if target_col else 'Skipped — no target selected.')}

## Feature Engineering
5 specific suggestions with actual column names and why.

## Modelling Readiness
Exact preprocessing steps remaining. Recommended algorithm and why. Any leakage risks by column name.

## Anomalies & Watch-outs
Anything unusual in the actual data. Impossible values, outliers, suspicious combinations.

Be specific. Use actual column names and values. Under 800 words."""

    return call_llm(prompt, provider, api_key, model)


# ── HTML Report ───────────────────────────────────────────────────────────────

def fig_to_b64(fig):
    buf = io.BytesIO()
    fig.write_image(buf, format="png", scale=2)
    return base64.b64encode(buf.getvalue()).decode()


def build_html_report(df, profile, summary, charts, target_col, is_aggregated):
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
    agg_badge     = "&nbsp;·&nbsp; <span style='background:#d4f5e9;color:#0a5c3e;padding:2px 8px;border-radius:4px;font-size:11px'>Aggregated data</span>" if is_aggregated else ""

    import re
    summary_html = re.sub(r"#{1,6}\s(.+)", r"<h3>\1</h3>", summary)
    summary_html = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", summary_html)
    summary_html = re.sub(r"`(.+?)`", r"<code>\1</code>", summary_html)
    summary_html = summary_html.replace("\n", "<br>")

    return f"""<!DOCTYPE html>
<html><head><meta charset="UTF-8">
<style>
  body {{ font-family: system-ui, sans-serif; max-width: 980px; margin: 40px auto; padding: 0 32px; color: #1a1a1a; line-height: 1.7; }}
  .cover {{ background: linear-gradient(135deg, #378ADD 0%, #185FA5 100%); border-radius: 16px; padding: 36px 40px; color: white; margin-bottom: 32px; }}
  .cover h1 {{ font-size: 26px; font-weight: 600; margin: 0 0 6px; color: white; }}
  .cover .meta {{ font-size: 13px; opacity: 0.8; }}
  .metrics {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin-bottom: 28px; }}
  .metric {{ background: #f7f7f7; border-radius: 10px; padding: 16px 18px; }}
  .metric-val {{ font-size: 28px; font-weight: 600; margin: 0; color: #1a1a1a; }}
  .metric-lbl {{ font-size: 12px; color: #888; margin: 4px 0 0; }}
  h2 {{ font-size: 17px; font-weight: 600; border-bottom: 2px solid #378ADD; padding-bottom: 6px; margin-top: 36px; color: #1a1a1a; }}
  h3 {{ font-size: 15px; font-weight: 500; margin-top: 20px; color: #333; }}
  .summary {{ font-size: 14px; line-height: 1.8; }}
  code {{ background: #f0f0f0; padding: 1px 6px; border-radius: 4px; font-size: 13px; }}
  ul {{ margin: 8px 0; padding-left: 20px; }}
  li {{ margin-bottom: 5px; font-size: 14px; }}
  .footer {{ text-align: center; font-size: 12px; color: #aaa; margin-top: 48px; border-top: 1px solid #eee; padding-top: 20px; }}
</style>
</head><body>
<div class="cover">
  <h1>📊 EDA Report</h1>
  <div class="meta">
    Generated {datetime.now().strftime("%d %B %Y, %H:%M")} &nbsp;·&nbsp;
    {df.shape[0]:,} rows &nbsp;·&nbsp; {df.shape[1]} columns{target_line}{agg_badge}
  </div>
</div>
<div class="metrics">
  <div class="metric"><p class="metric-val">{df.shape[0]:,}</p><p class="metric-lbl">Rows</p></div>
  <div class="metric"><p class="metric-val">{df.shape[1]}</p><p class="metric-lbl">Columns</p></div>
  <div class="metric"><p class="metric-val">{missing_total:,}</p><p class="metric-lbl">Missing values</p></div>
  <div class="metric"><p class="metric-val">{profile["duplicates"]}</p><p class="metric-lbl">Duplicates</p></div>
</div>
<h2>AI Analysis</h2>
<div class="summary">{summary_html}</div>
<h2>Visualizations</h2>
{charts_html}
<div class="footer">Generated by Auto EDA Tool &nbsp;·&nbsp; Built with Streamlit + Plotly</div>
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
    model = st.selectbox("Model", PROVIDERS[provider]["models"], label_visibility="collapsed")
    st.session_state.model = model
    api_key = st.text_input("API Key", type="password",
                            placeholder=PROVIDERS[provider]["key_hint"],
                            value=st.session_state.api_key,
                            label_visibility="collapsed")
    st.session_state.api_key = api_key

    key_links = {
        "Gemini": "https://aistudio.google.com",
        "OpenAI / ChatGPT": "https://platform.openai.com/api-keys",
        "Claude (Anthropic)": "https://console.anthropic.com",
        "Grok (xAI)": "https://console.x.ai",
    }
    if api_key:
        st.success(f"✓ {provider} key saved")
    else:
        st.caption(f"[Get free key →]({key_links[provider]})")

    st.divider()

    target_col = None
    do_clean   = False

    if uploaded:
        st.markdown("**③ Configure**")
        df_peek     = pd.read_csv(uploaded, nrows=5)
        uploaded.seek(0)
        col_options = ["— none —"] + list(df_peek.columns)
        target_sel  = st.selectbox("🎯 Target variable", col_options, label_visibility="collapsed")
        target_col  = None if target_sel == "— none —" else target_sel
        do_clean    = st.toggle("🧹 Auto-clean", value=False)

    st.divider()
    st.markdown("""
    <div style="font-size:11px;color:#aaa;line-height:2">
      🔵 numeric &nbsp; 🟢 categorical<br>
      🟣 datetime &nbsp; 🟡 binary &nbsp; ⚫ id
    </div>""", unsafe_allow_html=True)


# ── Landing ───────────────────────────────────────────────────────────────────

if not uploaded:
    st.markdown("""
    <div style="max-width:580px;margin:60px auto 0;text-align:center">
      <div style="font-size:48px;margin-bottom:16px">📊</div>
      <div style="font-size:26px;font-weight:600;margin-bottom:10px">Auto EDA Tool</div>
      <div style="font-size:15px;color:#666;margin-bottom:32px;line-height:1.6">
        Upload any CSV — raw or aggregated. The tool handles both.
        Aggregate first if needed, then get full EDA, time series,
        statistical tests, and AI-powered insights.
      </div>
      <div style="display:grid;grid-template-columns:1fr 1fr;gap:12px;text-align:left;margin-bottom:32px">
        <div style="background:#fff;border:1px solid #e8ecf0;border-radius:12px;padding:16px">
          <div style="font-size:22px;margin-bottom:6px">⚡</div>
          <div style="font-size:13px;font-weight:500;margin-bottom:4px">Smart aggregator</div>
          <div style="font-size:12px;color:#888">Group & sum raw transactional data before analysis</div>
        </div>
        <div style="background:#fff;border:1px solid #e8ecf0;border-radius:12px;padding:16px">
          <div style="font-size:22px;margin-bottom:6px">📅</div>
          <div style="font-size:13px;font-weight:500;margin-bottom:4px">Time series</div>
          <div style="font-size:12px;color:#888">Auto-detects dates, plots trends, seasonality & anomalies</div>
        </div>
        <div style="background:#fff;border:1px solid #e8ecf0;border-radius:12px;padding:16px">
          <div style="font-size:22px;margin-bottom:6px">🧪</div>
          <div style="font-size:13px;font-weight:500;margin-bottom:4px">Statistical tests</div>
          <div style="font-size:12px;color:#888">t-test, ANOVA, chi-square, correlation significance</div>
        </div>
        <div style="background:#fff;border:1px solid #e8ecf0;border-radius:12px;padding:16px">
          <div style="font-size:22px;margin-bottom:6px">🤖</div>
          <div style="font-size:13px;font-weight:500;margin-bottom:4px">AI + Chat</div>
          <div style="font-size:12px;color:#888">AI sees your actual data — ask follow-up questions</div>
        </div>
      </div>
      <div style="font-size:13px;color:#aaa">← Upload a CSV in the sidebar to begin</div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()


# ── Load raw data ─────────────────────────────────────────────────────────────

@st.cache_data
def load_raw(uploaded_file):
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
    return df

raw_df = load_raw(uploaded)


# ── Aggregator (step 0) ───────────────────────────────────────────────────────

st.markdown("""
<div style="background:#fff;border:1px solid #e8ecf0;border-radius:14px;padding:16px 20px;margin-bottom:20px">
  <div style="font-size:14px;font-weight:600;margin-bottom:4px">⚡ Step 1 — Aggregate your data (optional)</div>
  <div style="font-size:13px;color:#888">If your data is raw/transactional, group and sum it first. Skip if already aggregated.</div>
</div>
""", unsafe_allow_html=True)

with st.expander("⚡ Open aggregator", expanded=False):
    all_cols  = list(raw_df.columns)
    num_cols_raw = [c for c in raw_df.columns if pd.api.types.is_numeric_dtype(raw_df[c])]
    cat_cols_raw = [c for c in raw_df.columns if not pd.api.types.is_numeric_dtype(raw_df[c])]

    c1, c2, c3 = st.columns(3)
    group_cols = c1.multiselect("Group by", all_cols,
                                default=[cat_cols_raw[0]] if cat_cols_raw else [])
    agg_cols   = c2.multiselect("Aggregate (numeric)", num_cols_raw,
                                default=num_cols_raw[:2] if num_cols_raw else [])
    agg_func   = c3.selectbox("Method", ["sum", "mean", "count", "max", "min", "median"])

    if group_cols and agg_cols:
        agg_dict   = {c: agg_func for c in agg_cols}
        preview_df = raw_df.groupby(group_cols).agg(agg_dict).reset_index().round(2)
        preview_df.columns = group_cols + [f"{agg_func}_{c}" for c in agg_cols]
        st.dataframe(preview_df.head(10), use_container_width=True)
        st.caption(f"{len(preview_df):,} groups from {len(raw_df):,} raw rows")

        if st.button("✅ Use this aggregated data for analysis"):
            st.session_state.working_df    = preview_df.to_json(orient="split")
            st.session_state.is_aggregated = True
            st.success("Aggregated data loaded. Scroll down to see analysis.")
            st.rerun()

    if st.button("⏭ Skip — use raw data as-is"):
        st.session_state.working_df    = raw_df.to_json(orient="split")
        st.session_state.is_aggregated = False
        st.rerun()

if not st.session_state.working_df:
    st.info("Choose to aggregate or skip above to begin analysis.")
    st.stop()


# ── Working dataframe ─────────────────────────────────────────────────────────

working_df    = pd.read_json(io.StringIO(st.session_state.working_df), orient="split")
is_aggregated = st.session_state.is_aggregated
profile       = profile_df(st.session_state.working_df)


# ── Header bar ────────────────────────────────────────────────────────────────

fname    = uploaded.name
agg_tag  = "<span style='background:#d4f5e9;color:#0a5c3e;border-radius:20px;padding:3px 12px;font-size:11px;font-weight:500;margin-left:8px'>Aggregated</span>" if is_aggregated else ""
tgt_tag  = f"<span style='background:#d4eaff;color:#185FA5;border-radius:20px;padding:3px 12px;font-size:11px;font-weight:500;margin-left:8px'>🎯 {target_col}</span>" if target_col else ""

type_count = {}
for m in profile["columns"].values():
    type_count[m["type"]] = type_count.get(m["type"], 0) + 1
type_str = "  ·  ".join(f"{v} {k}" for k, v in type_count.items())

st.markdown(f"""
<div style="background:#fff;border:1px solid #e8ecf0;border-radius:14px;
     padding:14px 20px;margin-bottom:16px;display:flex;align-items:center;gap:12px">
  <div style="font-size:24px">📄</div>
  <div>
    <div style="font-size:14px;font-weight:500">{fname}{agg_tag}{tgt_tag}</div>
    <div style="font-size:12px;color:#888;margin-top:2px">{type_str}</div>
  </div>
  <div style="margin-left:auto">
    <span style="font-size:12px;color:#aaa;cursor:pointer"
          onclick="window.location.reload()">🔄 Reset</span>
  </div>
</div>
""", unsafe_allow_html=True)

m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Rows",       f"{profile['shape'][0]:,}")
m2.metric("Columns",    profile["shape"][1])
m3.metric("Missing",    f"{sum(profile['missing'].values()):,}")
m4.metric("Duplicates", profile["duplicates"])
m5.metric("Size",       f"{profile.get('memory_mb','?')} MB")

st.markdown("<div style='margin-bottom:4px'></div>", unsafe_allow_html=True)


# ── Tabs ──────────────────────────────────────────────────────────────────────

tab1, tab2, tab3 = st.tabs([
    "📊 Data & Charts",
    "🔬 Analysis",
    "🤖 AI + Report",
])


# ── Tab 1: Data & Charts ──────────────────────────────────────────────────────

with tab1:

    # Column profiles
    st.markdown("#### Column profiles")
    type_colors   = {"numeric":"🔵","categorical":"🟢","datetime":"🟣","binary":"🟡","id":"⚫"}
    total_missing = sum(profile["missing"].values())
    if total_missing > 0:
        st.markdown(f"""
        <div class="warn-box">
          ⚠️ <strong>{total_missing:,} missing values</strong> across
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
                c2.metric("Std",      round(meta["stats"]["std"],  3))
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
                st.info(f"ID column — {meta['unique']} unique ({meta['unique_pct']}% of rows). Excluded from modelling.")
            else:
                st.write(f"**Unique:** {meta['unique']} ({meta['unique_pct']}% of rows)")
                st.dataframe(pd.DataFrame.from_dict(meta["top5"], orient="index", columns=["count"]),
                             use_container_width=True)
            if meta["nulls"] > 0:
                st.warning(f"{meta['nulls']} missing ({meta['null_pct']}% of rows)")

    st.divider()

    # Auto charts
    st.markdown("#### Auto-generated charts")
    with st.spinner("Generating charts..."):
        charts = generate_charts(working_df, profile)

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
        st.markdown(f"**{section_title}**")
        left, right = st.columns(2)
        for i, (_, col, fig) in enumerate(section_charts):
            (left if i % 2 == 0 else right).plotly_chart(fig, use_container_width=True)

    # Auto-clean
    if do_clean:
        st.divider()
        st.markdown("#### 🧹 Auto-clean")
        with st.spinner("Cleaning..."):
            cleaned_df, clean_log = auto_clean(working_df, profile, target_col)
        ca, cb, cc = st.columns(3)
        ca.metric("Columns before", working_df.shape[1])
        cb.metric("Columns after",  cleaned_df.shape[1])
        cc.metric("Rows after",     f"{len(cleaned_df):,}")
        for entry in clean_log:
            st.markdown(f'<div class="insight-box">{entry}</div>', unsafe_allow_html=True)
        st.dataframe(cleaned_df.head(10), use_container_width=True)
        st.download_button("⬇️ Download cleaned CSV",
                           data=cleaned_df.to_csv(index=False).encode("utf-8"),
                           file_name="cleaned_data.csv", mime="text/csv")


# ── Tab 2: Analysis ───────────────────────────────────────────────────────────

with tab2:

    # Time series
    render_timeseries(working_df, profile)

    st.divider()

    # Statistical tests
    render_stat_tests(working_df, profile)

    # Target analysis
    if target_col:
        st.divider()
        st.markdown("#### 🎯 Target variable analysis")
        render_target(working_df, profile, target_col)
    else:
        st.divider()
        st.markdown("""
        <div style="text-align:center;padding:32px 0">
          <div style="font-size:32px;margin-bottom:8px">🎯</div>
          <div style="font-size:14px;font-weight:500;margin-bottom:4px">No target variable selected</div>
          <div style="font-size:13px;color:#888">Select a target column in the sidebar to see feature analysis.</div>
        </div>""", unsafe_allow_html=True)


# ── Tab 3: AI + Report ────────────────────────────────────────────────────────

with tab3:
    if not st.session_state.api_key:
        st.markdown("""
        <div style="text-align:center;padding:48px 0">
          <div style="font-size:36px;margin-bottom:12px">🤖</div>
          <div style="font-size:15px;font-weight:500;margin-bottom:6px">No API key yet</div>
          <div style="font-size:13px;color:#888">Add your API key in the sidebar.</div>
        </div>""", unsafe_allow_html=True)
    else:
        prov  = st.session_state.provider
        mdl   = st.session_state.model
        a_key = st.session_state.api_key

        st.markdown(f"""
        <div class="insight-box">
          Using <strong>{prov}</strong> / <code>{mdl}</code>
          {"&nbsp;·&nbsp; 🎯 Target: <strong>" + target_col + "</strong>" if target_col else ""}
          {"&nbsp;·&nbsp; Aggregated data" if is_aggregated else ""}
        </div>""", unsafe_allow_html=True)

        sample_json = working_df.to_json(orient="split")

        with st.spinner(f"Generating AI analysis with {prov}..."):
            try:
                summary = generate_summary(
                    profile, target_col, prov, a_key, mdl,
                    sample_json, is_aggregated,
                )
                st.markdown(summary)
                st.divider()

                with st.spinner("Building report..."):
                    html = build_html_report(
                        working_df, profile, summary, charts,
                        target_col, is_aggregated,
                    )
                st.download_button(
                    label="⬇️ Download full report (.html)",
                    data=html,
                    file_name="eda_report.html",
                    mime="text/html",
                )

                # ── Chat ──────────────────────────────────────────────────
                st.divider()
                st.markdown("### 💬 Ask about your data")
                st.caption("Ask follow-up questions. The AI has full context of your dataset.")

                if "chat_history" not in st.session_state:
                    st.session_state.chat_history = []

                if st.session_state.chat_history:
                    if st.button("Clear chat"):
                        st.session_state.chat_history = []
                        st.rerun()

                for msg in st.session_state.chat_history:
                    with st.chat_message(msg["role"]):
                        st.markdown(msg["content"])

                user_q = st.chat_input("Ask anything about your data...")

                if user_q:
                    st.session_state.chat_history.append({"role": "user", "content": user_q})
                    with st.chat_message("user"):
                        st.markdown(user_q)

                    sample_str = working_df.head(15).to_string()
                    col_types  = "\n".join(
                        f"- `{col}`: {meta['type']} | {meta['nulls']} nulls"
                        for col, meta in profile["columns"].items()
                    )
                    system_ctx = f"""You are a senior data scientist analyzing a dataset.
Dataset: {profile['shape'][0]:,} rows × {profile['shape'][1]} columns
Target: {target_col if target_col else 'none'}
Aggregated: {is_aggregated}

Columns:
{col_types}

Sample data:
{sample_str}

AI summary already generated:
{summary}

Answer questions about this dataset specifically. Reference actual column names and values.
Write clean Python/pandas code if asked. Be concise and specific."""

                    with st.chat_message("assistant"):
                        with st.spinner("Thinking..."):
                            try:
                                if prov == "Gemini":
                                    import google.generativeai as genai
                                    genai.configure(api_key=a_key)
                                    history_str = "\n\n".join(
                                        f"{m['role'].upper()}: {m['content']}"
                                        for m in st.session_state.chat_history[:-1]
                                    )
                                    full = f"{system_ctx}\n\n{history_str}\n\nUSER: {user_q}"
                                    answer = genai.GenerativeModel(mdl).generate_content(full).text

                                elif prov in ("OpenAI / ChatGPT", "Grok (xAI)"):
                                    from openai import OpenAI
                                    base = "https://api.x.ai/v1" if prov == "Grok (xAI)" else None
                                    client = OpenAI(api_key=a_key, **({"base_url": base} if base else {}))
                                    msgs = [{"role": "system", "content": system_ctx}]
                                    for m in st.session_state.chat_history[:-1]:
                                        msgs.append({"role": m["role"], "content": m["content"]})
                                    msgs.append({"role": "user", "content": user_q})
                                    answer = client.chat.completions.create(
                                        model=mdl, messages=msgs, max_tokens=1000
                                    ).choices[0].message.content

                                elif prov == "Claude (Anthropic)":
                                    import anthropic
                                    msgs = []
                                    for m in st.session_state.chat_history[:-1]:
                                        msgs.append({"role": m["role"], "content": m["content"]})
                                    msgs.append({"role": "user", "content": user_q})
                                    answer = anthropic.Anthropic(api_key=a_key).messages.create(
                                        model=mdl, max_tokens=1000,
                                        system=system_ctx, messages=msgs,
                                    ).content[0].text

                                st.markdown(answer)
                                st.session_state.chat_history.append(
                                    {"role": "assistant", "content": answer}
                                )
                            except Exception as e:
                                st.error(f"Error: {e}")

            except Exception as e:
                st.error(f"Error from {prov}: {e}")
