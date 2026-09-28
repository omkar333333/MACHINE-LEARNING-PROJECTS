# 🧠 RAG Document Assistant

A lightweight, modular Retrieval-Augmented Generation (RAG) system engineered in Python to ground queries against arbitrary documentation, lecture notes, or textbooks using vector similarity retrieval.

[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=flat&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)

---

## 🏗️ Architecture

```
[Uploaded Document (.txt / .md)]
             │
             ▼
   [Sentence/Paragraph Chunking]
             │
             ▼
    [Vector Indexing & Sparse TF-IDF]
             │
   User Query ──► [Cosine Similarity Matching]
                         │
                         ▼
             [Top-K Re-ranked Excerpts]
                         │
                         ▼
             [Grounded Prompt Synthesis]
```

---

## 🚀 Quickstart

```bash
# Navigate to project folder
cd RAG-Document-Assistant

# Launch web interface
streamlit run app.py
```

---

## 👨‍💻 Author

**Omkar Mote** — [github.com/omkar333333](https://github.com/omkar333333)
