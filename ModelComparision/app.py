# # Streamlit App: Compare Foundation vs Domain-Specific NLP Models for Fill-Mask Task

# import streamlit as st
# import pandas as pd
# from transformers import pipeline

# # --- UI Configuration ---
# st.set_page_config(
#     page_title="NLP Model Comparison",
#     layout="wide"
# )

# st.title("🎯 Foundation vs. Domain-Specific Model Comparison")
# st.write("""
# This app compares a general-purpose **foundation model** with a **domain-specific model** on a **Fill-Mask** task. 
# Enter a sentence with a `[MASK]` token and see which model provides more contextually relevant predictions.
# """)

# # --- Model Selection ---
# # Models suitable for the Fill-Mask task
# FOUNDATION_MODELS = {
#     "BERT (Base, Uncased)": "bert-base-uncased",
#     "RoBERTa (Base)": "roberta-base",
#     "DistilBERT (Base, Uncased)": "distilbert-base-uncased",
# }

# DOMAIN_MODELS = {
#     "BioClinicalBERT (Medical)": "emilyalsentzer/Bio_ClinicalBERT",
#     "SciBERT (Scientific)": "allenai/scibert_scivocab_uncased",
#     "FinBERT (Financial)": "ProsusAI/finbert",
#     "LegalBERT (Legal)": "nlpaueb/legal-bert-base-uncased",
# }

# # --- Sample Texts ---
# # Each sample now includes a [MASK] token for the fill-mask task.
# SAMPLE_TEXTS = {
#     "Medical": "A patient with chronic kidney disease requires weekly [MASK].",
#     "Scientific": "New research in [MASK] shows promising results for cancer treatment.",
#     "Financial": "The company's stock price saw a significant [MASK] after the earnings report.",
#     "Legal": "The defendant was found [MASK] by the jury.",
#     "General": "The best way to learn a new language is to practice it [MASK]."
# }

# # --- Caching ---
# # Cache the pipeline creation for performance
# @st.cache_resource(show_spinner="Loading models...")
# def load_fill_mask_pipeline(model_name):
#     """Loads a fill-mask pipeline for a given model name."""
#     return pipeline("fill-mask", model=model_name)

# # --- App Layout ---
# col1, col2 = st.columns(2)

# with col1:
#     st.header("Foundation Model")
#     foundation_choice_key = st.selectbox(
#         "Choose a foundation model:",
#         options=list(FOUNDATION_MODELS.keys())
#     )
#     foundation_model_name = FOUNDATION_MODELS[foundation_choice_key]

# with col2:
#     st.header("Domain-Specific Model")
#     domain_choice_key = st.selectbox(
#         "Choose a domain-specific model:",
#         options=list(DOMAIN_MODELS.keys())
#     )
#     domain_model_name = DOMAIN_MODELS[domain_choice_key]

# # --- Input Section ---
# st.header("Input Text")
# input_option = st.radio(
#     "Choose an input method:",
#     ["Select from samples", "Enter manually"],
#     horizontal=True,
#     label_visibility="collapsed"
# )

# if input_option == "Select from samples":
#     sample_category = st.selectbox("Select a domain:", list(SAMPLE_TEXTS.keys()))
#     input_text = st.text_area(
#         "Input sentence with `[MASK]` token:",
#         value=SAMPLE_TEXTS[sample_category],
#         height=100
#     )
# else:
#     input_text = st.text_area(
#         "Input sentence with `[MASK]` token:",
#         value="The patient's symptoms suggest a diagnosis of [MASK].",
#         height=100
#     )

# # --- Run & Display Results ---
# if st.button("🚀 Compare Models"):
#     if not input_text or "[MASK]" not in input_text:
#         st.error("Please provide an input sentence containing the `[MASK]` token.")
#     else:
#         # Load pipelines
#         pipe_foundation = load_fill_mask_pipeline(foundation_model_name)
#         pipe_domain = load_fill_mask_pipeline(domain_model_name)

#         # Get predictions
#         with st.spinner("Generating predictions..."):
#             predictions_foundation = pipe_foundation(input_text)
#             predictions_domain = pipe_domain(input_text)

#         st.header("📊 Comparison of Top Predictions")
#         st.info(f"**Your Input:** {input_text.replace('[MASK]', '**[MASK]**')}")

#         # Create DataFrames for display
#         df_foundation = pd.DataFrame(predictions_foundation).rename(
#             columns={"score": "Confidence", "token_str": "Predicted Token", "sequence": "Filled Sentence"}
#         )
#         df_foundation['Confidence'] = df_foundation['Confidence'].map('{:.2%}'.format)

#         df_domain = pd.DataFrame(predictions_domain).rename(
#             columns={"score": "Confidence", "token_str": "Predicted Token", "sequence": "Filled Sentence"}
#         )
#         df_domain['Confidence'] = df_domain['Confidence'].map('{:.2%}'.format)

#         # Display results side-by-side
#         res_col1, res_col2 = st.columns(2)
#         with res_col1:
#             st.subheader(f"Results from `{foundation_choice_key}`")
#             st.dataframe(df_foundation[['Predicted Token', 'Confidence', 'Filled Sentence']], use_container_width=True)

#         with res_col2:
#             st.subheader(f"Results from `{domain_choice_key}`")
#             st.dataframe(df_domain[['Predicted Token', 'Confidence', 'Filled Sentence']], use_container_width=True)


# Streamlit App: Compare Foundation vs Domain-Specific NLP Models for Fill-Mask Task

import streamlit as st
import pandas as pd
import plotly.express as px
from transformers import pipeline

# --- UI Configuration ---
st.set_page_config(
    page_title="NLP Model Comparison",
    layout="wide"
)

st.title("🎯 Foundation vs. Domain-Specific Model Comparison")
st.write("""
This app compares a general-purpose **foundation model** with a **domain-specific model** on a **Fill-Mask** task. 
Enter a sentence with a `[MASK]` token to see which model provides more contextually relevant predictions. The results are shown in both graphs and tables.
""")

# --- Model Selection ---
# Models suitable for the Fill-Mask task
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

# --- Sample Texts ---
# Each sample now includes a [MASK] token for the fill-mask task.
SAMPLE_TEXTS = {
    "Medical": "A patient with chronic kidney disease requires weekly [MASK].",
    "Scientific": "New research in [MASK] shows promising results for cancer treatment.",
    "Financial": "The company's stock price saw a significant [MASK] after the earnings report.",
    "Legal": "The defendant was found [MASK] by the jury.",
    "General": "The best way to learn a new language is to practice it [MASK]."
}

# --- Caching ---
# Cache the pipeline creation for performance
@st.cache_resource(show_spinner="Loading models...")
def load_fill_mask_pipeline(model_name):
    """Loads a fill-mask pipeline for a given model name."""
    return pipeline("fill-mask", model=model_name)

# --- App Layout ---
st.sidebar.header("⚙️ Model Selection")
foundation_choice_key = st.sidebar.selectbox(
    "Choose a Foundation Model:",
    options=list(FOUNDATION_MODELS.keys())
)
foundation_model_name = FOUNDATION_MODELS[foundation_choice_key]

domain_choice_key = st.sidebar.selectbox(
    "Choose a Domain-Specific Model:",
    options=list(DOMAIN_MODELS.keys())
)
domain_model_name = DOMAIN_MODELS[domain_choice_key]

# --- Input Section ---
st.header("✍️ Input Text")
input_option = st.radio(
    "Choose an input method:",
    ["Select from samples", "Enter manually"],
    horizontal=True,
    label_visibility="collapsed"
)

if input_option == "Select from samples":
    sample_category = st.selectbox("Select a sample sentence domain:", list(SAMPLE_TEXTS.keys()))
    input_text = st.text_area(
        "Input sentence with `[MASK]` token:",
        value=SAMPLE_TEXTS[sample_category],
        height=100
    )
else:
    input_text = st.text_area(
        "Input sentence with `[MASK]` token:",
        value="The patient's symptoms suggest a diagnosis of [MASK].",
        height=100
    )

# --- Run & Display Results ---
if st.button("🚀 Compare Models"):
    if not input_text or "[MASK]" not in input_text:
        st.error("Please provide an input sentence containing the `[MASK]` token.")
    else:
        # Load pipelines
        pipe_foundation = load_fill_mask_pipeline(foundation_model_name)
        pipe_domain = load_fill_mask_pipeline(domain_model_name)

        # Get predictions
        with st.spinner("Generating predictions..."):
            predictions_foundation = pipe_foundation(input_text)
            predictions_domain = pipe_domain(input_text)

        st.header("📊 Comparison of Top Predictions")
        st.info(f"**Your Input:** {input_text.replace('[MASK]', '**[MASK]**')}")

        col1, col2 = st.columns(2)

        # --- Display Foundation Model Results ---
        with col1:
            st.subheader(f"Results from `{foundation_choice_key}`")
            
            # Create a DataFrame from raw predictions
            df_foundation = pd.DataFrame(predictions_foundation)
            
            # 1. Create and display the interactive graph
            fig_foundation = px.bar(
                df_foundation,
                x="token_str",
                y="score",
                title=f"Top Predictions by {foundation_choice_key}",
                labels={"token_str": "Predicted Token", "score": "Confidence Score"},
                color="score",
                color_continuous_scale=px.colors.sequential.Blues_r,
                text_auto='.2%'
            )
            fig_foundation.update_layout(xaxis_title="", yaxis_title="Confidence")
            st.plotly_chart(fig_foundation, use_container_width=True)

            # 2. Prepare and display the formatted table
            df_foundation_display = df_foundation.rename(
                columns={"score": "Confidence", "token_str": "Predicted Token", "sequence": "Filled Sentence"}
            )
            df_foundation_display['Confidence'] = df_foundation_display['Confidence'].map('{:.2%}'.format)
            st.dataframe(df_foundation_display[['Predicted Token', 'Confidence', 'Filled Sentence']], use_container_width=True, hide_index=True)

        # --- Display Domain-Specific Model Results ---
        with col2:
            st.subheader(f"Results from `{domain_choice_key}`")
            
            # Create a DataFrame from raw predictions
            df_domain = pd.DataFrame(predictions_domain)

            # 1. Create and display the interactive graph
            fig_domain = px.bar(
                df_domain,
                x="token_str",
                y="score",
                title=f"Top Predictions by {domain_choice_key}",
                labels={"token_str": "Predicted Token", "score": "Confidence Score"},
                color="score",
                color_continuous_scale=px.colors.sequential.Greens_r,
                text_auto='.2%'
            )
            fig_domain.update_layout(xaxis_title="", yaxis_title="Confidence")
            st.plotly_chart(fig_domain, use_container_width=True)

            # 2. Prepare and display the formatted table
            df_domain_display = df_domain.rename(
                columns={"score": "Confidence", "token_str": "Predicted Token", "sequence": "Filled Sentence"}
            )
            df_domain_display['Confidence'] = df_domain_display['Confidence'].map('{:.2%}'.format)
            st.dataframe(df_domain_display[['Predicted Token', 'Confidence', 'Filled Sentence']], use_container_width=True, hide_index=True)