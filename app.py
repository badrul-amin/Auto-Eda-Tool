import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import anthropic
import json

st.set_page_config(page_title="Auto EDA", layout="wide")

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

@st.cache_data
def generate_charts(_df, profile):
    figs = []
    for col, meta in profile["columns"].items():
        fig, ax = plt.subplots(figsize=(5, 3))
        if meta["type"] == "numeric":
            sns.histplot(_df[col].dropna(), kde=True, ax=ax, color="#378ADD")
            ax.set_title(f"Distribution: {col}", fontsize=12)
        else:
            _df[col].value_counts().head(10).plot(kind="barh", ax=ax, color="#1D9E75")
            ax.set_title(f"Top values: {col}", fontsize=12)
        ax.set_xlabel("")
        plt.tight_layout()
        figs.append(fig)
        plt.close(fig)

    num_cols = [c for c, m in profile["columns"].items() if m["type"] == "numeric"]
    if len(num_cols) > 1:
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(_df[num_cols].corr().round(2), annot=True, fmt=".2f", cmap="Blues", ax=ax)
        ax.set_title("Correlation matrix", fontsize=12)
        plt.tight_layout()
        figs.append(fig)
        plt.close(fig)

    return figs

@st.cache_data
def generate_summary(profile, api_key):
    client = anthropic.Anthropic(api_key=api_key)
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

    msg = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
    )
    return msg.content[0].text


# ── UI ────────────────────────────────────────────────────────────────────────

st.title("Auto EDA Tool")
st.caption("Upload any CSV — get instant analysis + AI summary")

uploaded = st.file_uploader("Drop your CSV here", type=["csv"])

if uploaded:
    df, profile = profile_csv(uploaded)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Rows", f"{profile['shape'][0]:,}")
    m2.metric("Columns", profile["shape"][1])
    m3.metric("Missing values", f"{sum(profile['missing'].values()):,}")
    m4.metric("Duplicates", profile["duplicates"])

    st.divider()
    tab1, tab2, tab3 = st.tabs(["Columns", "Charts", "AI Summary"])

    with tab1:
        for col, meta in profile["columns"].items():
            warn = f"  ⚠ {meta['nulls']} missing" if meta["nulls"] > 0 else ""
            with st.expander(f"**{col}**  ·  {meta['type']}{warn}"):
                if meta["type"] == "numeric":
                    st.dataframe(pd.DataFrame(meta["stats"], index=["value"]).T)
                else:
                    st.write(f"Unique values: {meta['unique']}")
                    st.dataframe(pd.DataFrame.from_dict(meta["top5"], orient="index", columns=["count"]))

    with tab2:
        figs = generate_charts(df, profile)
        left, right = st.columns(2)
        for i, fig in enumerate(figs):
            (left if i % 2 == 0 else right).pyplot(fig)

    with tab3:
        api_key = st.text_input("Anthropic API key", type="password", placeholder="sk-ant-...")
        if api_key:
            with st.spinner("Claude is analysing your data..."):
                try:
                    summary = generate_summary(profile, api_key)
                    st.markdown(summary)
                    st.download_button("Download summary", data=summary, file_name="eda_summary.txt")
                except Exception as e:
                    st.error(f"API error: {e}")
        else:
            st.info("Enter your Anthropic API key above to generate the AI summary.")
