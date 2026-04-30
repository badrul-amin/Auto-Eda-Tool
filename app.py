import json
import base64
import io

import streamlit as st
import pandas as pd
import plotly.express as px
import google.generativeai as genai

st.set_page_config(page_title="Auto EDA", layout="wide")

API_KEY = "AIzaSyA0eLwOItEMlrKPekag2LWWCBsDCSRYt1Y"  # paste your Gemini key here


# ── Profiler ──────────────────────────────────────────────────────────────────

@st.cache_data
def profile_csv(uploaded_file) -> tuple:
    df = pd.read_csv(uploaded_file)
    profile = {
        "shape": list(df.shape),
        "columns": {},
        "missing": df.isnull().sum().to_dict(),
        "duplicates": int(df.duplicated().sum()),
    }
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            profile["columns"][col] = {
                "type": "numeric",
                "stats": df[col].describe().round(2).to_dict(),
                "nulls": int(df[col].isnull().sum()),
            }
        else:
            profile["columns"][col] = {
                "type": "categorical",
                "unique": int(df[col].nunique()),
                "top5": df[col].value_counts().head(5).to_dict(),
                "nulls": int(df[col].isnull().sum()),
            }
    return df, profile


# ── Charts ────────────────────────────────────────────────────────────────────

def clean_layout():
    return dict(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="system-ui, sans-serif", size=12, color="#555"),
        title=dict(font=dict(size=14), x=0),
        margin=dict(l=0, r=0, t=40, b=0),
        xaxis=dict(showgrid=True, gridcolor="#f0f0f0", zeroline=False),
        yaxis=dict(showgrid=False, zeroline=False),
        showlegend=False,
        height=280,
    )


@st.cache_data
def generate_charts(_df, profile):
    charts = []
    num_cols = [c for c, m in profile["columns"].items() if m["type"] == "numeric"]
    cat_cols = [c for c, m in profile["columns"].items() if m["type"] == "categorical"]

    for col in num_cols:
        fig = px.histogram(
            _df[col].dropna(),
            nbins=30,
            title=f"Distribution: {col}",
            color_discrete_sequence=["#378ADD"],
        )
        fig.update_layout(**clean_layout())
        fig.update_traces(marker_line_width=0)
        charts.append(("numeric", col, fig))

    for col in cat_cols:
        top = _df[col].value_counts().head(10).reset_index()
        top.columns = [col, "count"]
        fig = px.bar(
            top, x="count", y=col, orientation="h",
            title=f"Top values: {col}",
            color_discrete_sequence=["#1D9E75"],
        )
        fig.update_layout(**clean_layout())
        fig.update_traces(marker_line_width=0)
        charts.append(("categorical", col, fig))

    if len(num_cols) > 1:
        corr = _df[num_cols].corr().round(2)
        fig = px.imshow(
            corr,
            text_auto=True,
            color_continuous_scale="Blues",
            title="Correlation matrix",
            aspect="auto",
        )
        fig.update_layout(**clean_layout())
        charts.append(("corr", "correlation", fig))

    return charts


# ── LLM Summary ───────────────────────────────────────────────────────────────

@st.cache_data
def generate_summary(profile):
    genai.configure(api_key=API_KEY)
    model = genai.GenerativeModel("gemini-1.5-flash")
    prompt = f"""You are a data analyst writing for a non-technical stakeholder.

Dataset profile:
{json.dumps(profile, indent=2)}

Write a plain-English EDA summary covering:
1. Dataset shape and column types
2. Key patterns in numeric columns
3. Notable categorical breakdowns
4. Data quality issues (nulls, duplicates)
5. 2-3 actionable next steps

Be concise and specific. Use markdown formatting."""

    response = model.generate_content(prompt)
    return response.text


# ── HTML Report ───────────────────────────────────────────────────────────────

def fig_to_b64(fig):
    buf = io.BytesIO()
    fig.write_image(buf, format="png", scale=2)
    return base64.b64encode(buf.getvalue()).decode()


def build_html_report(df, profile, summary, charts):
    charts_html = ""
    for i in range(0, len(charts), 2):
        charts_html += '<div style="display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-bottom:16px">'
        for _, col, fig in charts[i:i+2]:
            charts_html += f'<img src="data:image/png;base64,{fig_to_b64(fig)}" style="width:100%;border-radius:8px">'
        charts_html += "</div>"

    missing_total = sum(profile["missing"].values())

    html = f"""<!DOCTYPE html>
<html><head><meta charset="UTF-8">
<style>
  body {{ font-family: system-ui, sans-serif; max-width: 900px; margin: 40px auto; padding: 0 24px; color: #1a1a1a; }}
  h1 {{ font-size: 22px; font-weight: 500; margin-bottom: 4px; }}
  .meta {{ font-size: 13px; color: #888; margin-bottom: 24px; }}
  .metrics {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin-bottom: 28px; }}
  .metric {{ background: #f5f5f5; border-radius: 8px; padding: 14px 16px; }}
  .metric-val {{ font-size: 24px; font-weight: 500; margin: 0; }}
  .metric-lbl {{ font-size: 12px; color: #888; margin: 4px 0 0; }}
  h2 {{ font-size: 15px; font-weight: 500; border-bottom: 1px solid #eee; padding-bottom: 8px; margin-top: 32px; }}
  .summary {{ font-size: 14px; line-height: 1.8; background: #f9f9f9; padding: 16px 20px; border-radius: 8px; white-space: pre-wrap; }}
</style>
</head><body>
<h1>EDA Report</h1>
<div class="meta">Generated by Auto EDA Tool &nbsp;·&nbsp; {df.shape[0]:,} rows &nbsp;·&nbsp; {df.shape[1]} columns</div>

<div class="metrics">
  <div class="metric"><p class="metric-val">{df.shape[0]:,}</p><p class="metric-lbl">Rows</p></div>
  <div class="metric"><p class="metric-val">{df.shape[1]}</p><p class="metric-lbl">Columns</p></div>
  <div class="metric"><p class="metric-val">{missing_total:,}</p><p class="metric-lbl">Missing values</p></div>
  <div class="metric"><p class="metric-val">{profile["duplicates"]}</p><p class="metric-lbl">Duplicates</p></div>
</div>

<h2>AI Summary</h2>
<div class="summary">{summary}</div>

<h2>Visualizations</h2>
{charts_html}
</body></html>"""

    return html


# ── UI ────────────────────────────────────────────────────────────────────────

st.title("Auto EDA Tool")
st.caption("Upload any CSV — get instant analysis + downloadable AI report")

uploaded = st.file_uploader("Drop your CSV here", type=["csv"])

if uploaded:
    df, profile = profile_csv(uploaded)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Rows", f"{profile['shape'][0]:,}")
    m2.metric("Columns", profile["shape"][1])
    m3.metric("Missing values", f"{sum(profile['missing'].values()):,}")
    m4.metric("Duplicates", profile["duplicates"])

    st.divider()
    tab1, tab2, tab3 = st.tabs(["Columns", "Charts", "AI Summary + Report"])

    with tab1:
        for col, meta in profile["columns"].items():
            warn = f"  ⚠ {meta['nulls']} missing" if meta["nulls"] > 0 else ""
            with st.expander(f"**{col}**  ·  {meta['type']}{warn}"):
                if meta["type"] == "numeric":
                    st.dataframe(
                        pd.DataFrame(meta["stats"], index=["value"]).T,
                        use_container_width=True,
                    )
                else:
                    st.write(f"Unique values: {meta['unique']}")
                    st.dataframe(
                        pd.DataFrame.from_dict(
                            meta["top5"], orient="index", columns=["count"]
                        ),
                        use_container_width=True,
                    )

    with tab2:
        charts = generate_charts(df, profile)
        left, right = st.columns(2)
        for i, (_, col, fig) in enumerate(charts):
            (left if i % 2 == 0 else right).plotly_chart(fig, use_container_width=True)

    with tab3:
        with st.spinner("Analysing your data..."):
            try:
                summary = generate_summary(profile)
                st.markdown(summary)
                st.divider()
                with st.spinner("Building report..."):
                    charts = generate_charts(df, profile)
                    html = build_html_report(df, profile, summary, charts)
                st.download_button(
                    label="Download report (.html)",
                    data=html,
                    file_name="eda_report.html",
                    mime="text/html",
                )
            except Exception as e:
                st.error(f"Error: {e}")
