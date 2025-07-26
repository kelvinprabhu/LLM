import streamlit as st
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import plotly.graph_objs as go
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from io import BytesIO
import time

st.set_page_config(page_title="🧠 Advanced Story Generator", layout="wide")

st.title("🧠 Advanced Story Generator & Model Comparator")

# Default prompts
default_prompts = [
    "In a world where dragons exist,",
    "A robot wakes up with emotions one morning.",
    "The kingdom fell into darkness when the sun vanished.",
    "A child finds a map to a hidden universe.",
    "Every mirror started reflecting the future instead of the present."
]

# Sidebar: user settings
st.sidebar.title("⚙️ Configuration")
prompts = st.sidebar.text_area("Enter prompts (one per line):", "\n".join(default_prompts)).splitlines()
max_len = st.sidebar.slider("Max story length", 100, 500, 250)

# Hugging Face models to compare
model_info = {
    "TinyLLaMA": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "GPT-2": "gpt2-medium",
    "GPT-Neo": "EleutherAI/gpt-neo-1.3B",
    "BART": "facebook/bart-large-cnn",
}

selected_models = st.sidebar.multiselect("Choose models:", model_info.keys(), default=list(model_info.keys()))

# Start generation
if st.sidebar.button("🚀 Generate Stories"):
    results = {}
    generation_times = {}

    device = "cuda" if torch.cuda.is_available() else "cpu"
    st.sidebar.success(f"Using device: {device.upper()}")

    for model_name in selected_models:
        st.subheader(f"📦 Loading: {model_name}")
        with st.spinner(f"Loading {model_name}..."):
            tokenizer = AutoTokenizer.from_pretrained(model_info[model_name])
            model = AutoModelForCausalLM.from_pretrained(
                model_info[model_name],
                torch_dtype=torch.float16 if device == "cuda" else torch.float32
            ).to(device)

            generator = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                device=0 if device == "cuda" else -1
            )

        stories = []
        total_time = 0
        for prompt in prompts:
            start = time.time()
            output = generator(prompt, max_length=max_len, do_sample=True, top_p=0.95, top_k=50)[0]['generated_text']
            total_time += time.time() - start
            stories.append(output)

        results[model_name] = stories
        generation_times[model_name] = round(total_time / len(prompts), 2)

    # 📜 Display generated stories
    st.header("📜 Generated Stories")
    for i, prompt in enumerate(prompts):
        st.markdown(f"### ✨ Prompt {i+1}: {prompt}")
        cols = st.columns(len(selected_models))
        for j, model in enumerate(selected_models):
            with cols[j]:
                st.markdown(f"**{model}**")
                st.text_area(label="", value=results[model][i], height=150)

    # 📊 Story Length Plot (Plotly)
    st.header("📈 Story Length Comparison")
    fig = go.Figure()
    for model in selected_models:
        lengths = [len(story.split()) for story in results[model]]
        fig.add_trace(go.Scatter(x=list(range(1, len(prompts)+1)), y=lengths, mode='lines+markers', name=model))
    fig.update_layout(title="Story Length per Prompt", xaxis_title="Prompt Index", yaxis_title="Word Count")
    st.plotly_chart(fig, use_container_width=True)

    # ⏱️ Avg generation time (Plotly)
    st.header("⏱️ Average Generation Time per Prompt")
    time_fig = go.Figure([go.Bar(
        x=list(generation_times.keys()), 
        y=list(generation_times.values()), 
        marker_color='indigo'
    )])
    time_fig.update_layout(title="Average Time per Prompt", yaxis_title="Seconds")
    st.plotly_chart(time_fig, use_container_width=True)

    # ☁️ WordClouds
    st.header("☁️ WordClouds")
    cols_wc = st.columns(len(selected_models))
    for idx, model in enumerate(selected_models):
        all_text = " ".join(results[model])
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_text)
        buf = BytesIO()
        plt.figure(figsize=(8, 4))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis("off")
        plt.tight_layout(pad=0)
        plt.savefig(buf, format="png")
        plt.close()
        cols_wc[idx].image(buf.getvalue(), caption=model)

    # 📋 Evaluation table
    st.header("📋 Creativity Evaluation Table")
    table = "| Model | Prompt | Story Start | Length | Creativity Score (1–5) |\n"
    table += "|-------|--------|-------------|--------|-------------------------|\n"
    for model in selected_models:
        for i, story in enumerate(results[model]):
            snippet = story[:60].replace('\n', ' ') + "..."
            table += f"| {model} | {i+1} | {snippet} | {len(story.split())} |     |\n"
    st.markdown(f"```\n{table}\n```")
