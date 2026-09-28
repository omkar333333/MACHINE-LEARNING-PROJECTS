"""
RAG Engine: Document Ingestion, Chunking, Vector Indexing, and Retrieval
Developed by: Omkar Mote (https://github.com/omkar333333)
"""

import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class LocalRAGEngine:
    def __init__(self, chunk_size=300, chunk_overlap=50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.chunks = []
        self.vectorizer = None
        self.chunk_vectors = None

    def chunk_text(self, text: str) -> list[str]:
        # Split text by sentence boundaries or paragraphs
        paragraphs = [p.strip() for p in text.split("\n") if len(p.strip()) > 0]
        chunks = []
        current_chunk = ""

        for para in paragraphs:
            if len(current_chunk) + len(para) < self.chunk_size:
                current_chunk += " " + para
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = para

        if current_chunk:
            chunks.append(current_chunk.strip())

        return chunks if chunks else [text]

    def ingest_document(self, text: str):
        self.chunks = self.chunk_text(text)
        if not self.chunks:
            return 0
        
        self.vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
        self.chunk_vectors = self.vectorizer.fit_transform(self.chunks)
        return len(self.chunks)

    def retrieve(self, query: str, top_k: int = 3) -> list[dict]:
        if not self.chunks or self.vectorizer is None:
            return []

        query_vec = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vec, self.chunk_vectors)[0]
        
        top_indices = np.argsort(similarities)[::-1][:top_k]
        results = []
        for idx in top_indices:
            score = float(similarities[idx])
            if score > 0.01:  # Relevance threshold
                results.append({
                    "chunk": self.chunks[idx],
                    "similarity": round(score, 4),
                    "chunk_id": int(idx)
                })

        return results

    def generate_context_prompt(self, query: str, top_k: int = 3) -> tuple[str, list[dict]]:
        retrieved = self.retrieve(query, top_k=top_k)
        if not retrieved:
            context = "No directly relevant context was located in the uploaded documentation."
        else:
            context = "\n---\n".join([f"[Source Chunk #{r['chunk_id']} | Score: {r['similarity']}]:\n{r['chunk']}" for r in retrieved])

        prompt = f"""You are an AI Document Assistant. Use the verified context excerpts below to answer the user query accurately.

Context Excerpts:
{context}

User Query:
{query}

Answer:"""
        return prompt, retrieved
