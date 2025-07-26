# streamlit_app.py
import streamlit as st
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import matplotlib.pyplot as plt
import time
import os

st.set_page_config(page_title="Creative Story Generator", layout="wide")
st.title("🧠 Creative Story Generator using Hugging Face Models")

# Sidebar configuration
st.sidebar.title("⚙️ Settings")
default_prompts = [
    "In a world where dragons exist,",
    "A robot wakes up with emotions one morning.",
    "The kingdom fell into darkness when the sun vanished.",
    "A child finds a map to a hidden universe.",
    "Every mirror started reflecting the future instead of the present."
]
prompts = st.sidebar.text_area("Enter prompts (one per line):", "\n".join(default_prompts)).splitlines()

model_info = {
    "tinyllama": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "gpt2": "gpt2-medium",
    "gptneo": "EleutherAI/gpt-neo-1.3B",
    "bart": "facebook/bart-large-cnn",
}

selected_models = st.sidebar.multiselect("Select models to compare:", model_info.keys(), default=list(model_info.keys()))
max_len = st.sidebar.slider("Max length of story:", 100, 500, 250)

if st.sidebar.button("Generate Stories"):
    results = {}
    generation_times = {}

    for model_id in selected_models:
        st.subheader(f"🔄 Loading model: {model_id}")
        with st.spinner(f"Loading {model_id}..."):
            tokenizer = AutoTokenizer.from_pretrained(model_info[model_id])
            model = AutoModelForCausalLM.from_pretrained(
                model_info[model_id],
                device_map="auto",
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
            )
            text_pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

        stories = []
        total_time = 0

        for prompt in prompts:
            start = time.time()
            output = text_pipe(prompt, max_length=max_len, num_return_sequences=1, do_sample=True, top_p=0.95, top_k=50)
            end = time.time()
            generated_text = output[0]['generated_text']
            stories.append(generated_text)
            total_time += (end - start)

        results[model_id] = stories
        generation_times[model_id] = total_time / len(prompts)

    # Show results
    st.header("📜 Generated Stories")
    for i, prompt in enumerate(prompts):
        st.markdown(f"### Prompt {i+1}: {prompt}")
        cols = st.columns(len(selected_models))
        for idx, model_id in enumerate(selected_models):
            with cols[idx]:
                st.markdown(f"**{model_id}**")
                st.text_area("", results[model_id][i], height=200)

    # Plot story lengths
    st.header("📊 Story Length Comparison")
    story_lengths = {model: [len(s.split()) for s in results[model]] for model in selected_models}
    fig1, ax1 = plt.subplots()
    for model in selected_models:
        ax1.plot(story_lengths[model], marker='o', label=model)
    ax1.set_title("Story Lengths Across Models")
    ax1.set_xlabel("Prompt Index")
    ax1.set_ylabel("Story Length (words)")
    ax1.legend()
    st.pyplot(fig1)

    # Plot generation time
    st.header("⏱️ Average Generation Time")
    fig2, ax2 = plt.subplots()
    ax2.bar(generation_times.keys(), generation_times.values(), color='orchid')
    ax2.set_title("Average Time per Prompt")
    ax2.set_ylabel("Seconds")
    st.pyplot(fig2)

    # Evaluation Table
    st.header("📋 Evaluation Table (Manual Scoring)")
    table = "| Model | Prompt | Story Start | Length | Human Creativity Score (1–5) |\n"
    table += "|-------|--------|-------------|--------|-------------------------------|\n"
    for model in selected_models:
        for i, story in enumerate(results[model]):
            snippet = story[:60].replace('\n', ' ') + "..."
            table += f"| {model} | {i+1} | {snippet} | {len(story.split())} |     |\n"
    st.markdown(f"```\n{table}\n```")
