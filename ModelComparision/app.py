import streamlit as st
import pandas as pd
import plotly.express as px
from transformers import pipeline

# --- UI Configuration ---
st.set_page_config(page_title="NLP Model Comparator", layout="wide", page_icon="🧠")

st.markdown("""
    <style>
        .main {background-color: #f8f9fa;}
        .block-container {padding: 2rem 2rem;}
        h1, h2, h3, h4 {color: #2c3e50;}
        .stButton>button {background-color: #1abc9c; color: white;}
        .stButton>button:hover {background-color: #16a085;}
        .stRadio>div>label {font-weight: bold;}
        .stDataFrameContainer {background-color: white; border-radius: 8px; box-shadow: 0 0 10px rgba(0,0,0,0.05);}
    </style>
""", unsafe_allow_html=True)

st.title("🧠 NLP Model Comparator: Foundation vs Domain-Specific")
st.write("Compare the performance of a general-purpose **foundation model** and a **domain-specific model** on a `Fill-Mask` task. This tool visualizes top predictions based on model confidence.")

# --- Model Lists ---
FOUNDATION_MODELS = {
    "BERT (Base, Uncased)": "bert-base-uncased",
    "RoBERTa (Base)": "roberta-base",
    "DistilBERT (Base, Uncased)": "distilbert-base-uncased",
}

DOMAIN_MODELS = {
    "BioClinicalBERT (Medical)": "emilyalsentzer/Bio_ClinicalBERT",
    "SciBERT (Scientific)": "allenai/scibert_scivocab_uncased",
    "FinBERT (Financial)": "ProsusAI/finbert",
    "LegalBERT (Legal)": "nlpaueb/legal-bert-base-uncased",
}

SAMPLE_TEXTS = {
    "Medical": "A patient with chronic kidney disease requires weekly [MASK].",
    "Scientific": "New research in [MASK] shows promising results for cancer treatment.",
    "Financial": "The company's stock price saw a significant [MASK] after the earnings report.",
    "Legal": "The defendant was found [MASK] by the jury.",
    "General": "The best way to learn a new language is to practice it [MASK]."
}

@st.cache_resource(show_spinner="🔄 Loading model...")
def load_fill_mask_pipeline(model_name):
    return pipeline("fill-mask", model=model_name)

# --- Sidebar Selection ---
st.sidebar.header("⚙️ Model Settings")

foundation_choice = st.sidebar.selectbox("Foundation Model", list(FOUNDATION_MODELS.keys()))
domain_choice = st.sidebar.selectbox("Domain-Specific Model", list(DOMAIN_MODELS.keys()))

foundation_model_name = FOUNDATION_MODELS[foundation_choice]
domain_model_name = DOMAIN_MODELS[domain_choice]

# --- Input Section ---
st.markdown("### ✍️ Input your masked sentence")

input_method = st.radio("Input Method:", ["Select a Sample", "Write Your Own"], horizontal=True)

if input_method == "Select a Sample":
    category = st.selectbox("Sample Domain", list(SAMPLE_TEXTS.keys()))
    input_text = st.text_area("Input Sentence", value=SAMPLE_TEXTS[category], height=100)
else:
    input_text = st.text_area("Enter sentence (use [MASK])", value="The patient's symptoms suggest a diagnosis of [MASK].", height=100)

if st.button("🚀 Run Comparison"):
    if "[MASK]" not in input_text:
        st.error("⚠️ Please include a `[MASK]` token in the input text.")
    else:
        with st.spinner("🔍 Generating predictions..."):
            pipe_foundation = load_fill_mask_pipeline(foundation_model_name)
            pipe_domain = load_fill_mask_pipeline(domain_model_name)

            predictions_foundation = pipe_foundation(input_text)
            predictions_domain = pipe_domain(input_text)

        st.markdown(f"### 📥 Input Provided")
        st.markdown(f"`{input_text}`")

        col1, col2 = st.columns(2)

        # --- Foundation Model Output ---
        with col1:
            st.markdown(f"### 🧱 Foundation Model: `{foundation_choice}`")
            df_f = pd.DataFrame(predictions_foundation)
            fig_f = px.bar(df_f, x="token_str", y="score", title=f"{foundation_choice} Predictions",
                           labels={"token_str": "Token", "score": "Confidence"},
                           color="score", text_auto='.2%',
                           color_continuous_scale="Blues")
            fig_f.update_layout(xaxis_title="", yaxis_title="Score", template="plotly_white")
            st.plotly_chart(fig_f, use_container_width=True)

            df_f_display = df_f.rename(columns={"token_str": "Prediction", "score": "Confidence", "sequence": "Completed Sentence"})
            df_f_display['Confidence'] = df_f_display['Confidence'].map('{:.2%}'.format)
            st.dataframe(df_f_display[["Prediction", "Confidence", "Completed Sentence"]], use_container_width=True, hide_index=True)

        # --- Domain Model Output ---
        with col2:
            st.markdown(f"### 🧩 Domain-Specific Model: `{domain_choice}`")
            df_d = pd.DataFrame(predictions_domain)
            fig_d = px.bar(df_d, x="token_str", y="score", title=f"{domain_choice} Predictions",
                           labels={"token_str": "Token", "score": "Confidence"},
                           color="score", text_auto='.2%',
                           color_continuous_scale="Greens")
            fig_d.update_layout(xaxis_title="", yaxis_title="Score", template="plotly_white")
            st.plotly_chart(fig_d, use_container_width=True)

            df_d_display = df_d.rename(columns={"token_str": "Prediction", "score": "Confidence", "sequence": "Completed Sentence"})
            df_d_display['Confidence'] = df_d_display['Confidence'].map('{:.2%}'.format)
            st.dataframe(df_d_display[["Prediction", "Confidence", "Completed Sentence"]], use_container_width=True, hide_index=True)
