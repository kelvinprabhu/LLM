# 🧠 Foundation vs. Domain-Specific NLP Model Comparison

> **Compare transformer models for masked language modeling in real-time using an interactive Streamlit interface.**

---

## 📌 Project Overview

This project provides an interactive platform to compare general-purpose **foundation models** like BERT and RoBERTa with **domain-specific transformers** like BioClinicalBERT and LegalBERT. Users can enter sentences containing `[MASK]` tokens and visualize how different models interpret them — useful for NLP practitioners making domain-driven model decisions.

---

## ✨ Key Features

- ✅ **Foundation vs Domain Model Comparison**

  - Run both models simultaneously on the same input with `[MASK]` and analyze outputs.

- ✍️ **Flexible Input Options**

  - Choose from predefined sample texts (Medical, Financial, etc.) or write your own input.

- 📊 **Interactive Graphs**

  - Plotly-based bar charts visualize prediction confidence for each token.

- 🧾 **Readable Data Tables**

  - See predicted tokens, confidence scores, and reconstructed sentences.

- ⚡ **Performance-Optimized**
  - Streamlit resource caching ensures models are loaded only once per session.

---

## 🧰 Technologies Used

- **Language:** Python 3.8+
- **Framework:** Streamlit
- **NLP Models:** Hugging Face Transformers
- **Visualization:** Plotly, Pandas

---

## 🛠 Installation Instructions

```bash
# 1. Clone the repository
git clone https://github.com/your-username/foundation-vs-domain-nlp.git
cd foundation-vs-domain-nlp

# 2. (Optional) Create a virtual environment
python -m venv venv
source venv/bin/activate  # or venv\\Scripts\\activate on Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the Streamlit app
streamlit run app.py
```
