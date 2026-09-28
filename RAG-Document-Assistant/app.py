"""
Streamlit Web UI for RAG Document Assistant
Developed by: Omkar Mote (https://github.com/omkar333333)
"""

import streamlit as st
from rag_engine import LocalRAGEngine

st.set_page_config(
    page_title="RAG Document Assistant | Omkar Mote",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 Retrieval-Augmented Generation (RAG) Assistant")
st.caption("Upload documents, perform semantic/keyword vector retrieval, and inspect synthesized context with citations.")

if "rag" not in st.session_state:
    st.session_state.rag = LocalRAGEngine()
if "doc_loaded" not in st.session_state:
    st.session_state.doc_loaded = False
if "chunk_count" not in st.session_state:
    st.session_state.chunk_count = 0

with st.sidebar:
    st.header("📄 Document Ingestion")
    uploaded_file = st.file_uploader("Upload Notes or Text File (.txt, .md)", type=["txt", "md"])
    sample_btn = st.button("Load Sample ML Notes")

    if sample_btn:
        sample_text = """
Machine Learning is a subfield of artificial intelligence that focuses on building applications that learn from data and improve their accuracy over time without being programmed to do so.
In supervised learning, algorithms are trained using labeled datasets where the input corresponds to a known output. Common supervised learning algorithms include Linear Regression, Logistic Regression, Support Vector Machines (SVM), Decision Trees, and Random Forests.
Unsupervised learning involves training models on unlabeled datasets. The system tries to learn patterns and the structure from data without human intervention. Principal Component Analysis (PCA) and K-Means clustering are prominent unsupervised techniques.
K-Means clustering partitions n observations into k clusters in which each observation belongs to the cluster with the nearest mean. The optimal number of clusters is often determined using the Elbow Method or Silhouette Analysis.
Random Forest is an ensemble learning method that constructs a multitude of decision trees at training time and outputs the mode of classes for classification or mean prediction for regression.
Overfitting occurs when a model learns the detail and noise in the training data to the extent that it negatively impacts the performance of the model on new data. Common regularization methods include L1 (Lasso) and L2 (Ridge) penalties.
"""
        count = st.session_state.rag.ingest_document(sample_text)
        st.session_state.doc_loaded = True
        st.session_state.chunk_count = count
        st.success(f"Sample notes indexed: {count} chunks!")

    elif uploaded_file is not None:
        raw_text = uploaded_file.read().decode("utf-8", errors="ignore")
        count = st.session_state.rag.ingest_document(raw_text)
        st.session_state.doc_loaded = True
        st.session_state.chunk_count = count
        st.success(f"File processed: {count} chunks indexed!")

    st.markdown("---")
    top_k = st.slider("Top-K Retrieved Chunks:", min_value=1, max_value=5, value=2)

if not st.session_state.doc_loaded:
    st.info("👈 Please upload a document or click **'Load Sample ML Notes'** in the sidebar to begin querying.")
else:
    st.success(f"✅ Active Vector Index contains **{st.session_state.chunk_count} document chunks**.")
    
    query = st.text_input("Ask a question about the document:", placeholder="e.g. How does K-Means clustering find the optimal number of clusters?")
    
    if query:
        prompt, retrieved = st.session_state.rag.generate_context_prompt(query, top_k=top_k)
        
        col_res, col_prompt = st.columns([3, 2])
        
        with col_res:
            st.subheader("📑 Top Retrieved Context Chunks")
            if not retrieved:
                st.warning("No context chunks met the similarity threshold for this query.")
            else:
                for idx, item in enumerate(retrieved, 1):
                    with st.expander(f"Chunk #{item['chunk_id']} — Similarity Score: {item['similarity']:.4f}", expanded=True):
                        st.markdown(f"> {item['chunk']}")
        
        with col_prompt:
            st.subheader("🤖 Synthesized Prompt Payload")
            st.caption("This prompt is ready to be passed to LLMs (OpenAI, Gemini, Ollama) for grounded response generation:")
            st.code(prompt, language="markdown")
