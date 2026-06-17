"""
embed_kb.py
Creates embeddings for haw_kb_en.json and haw_kb_de.json
and stores them as FAISS vector indices.
"""
import streamlit as st
import json
import faiss
import numpy as np
from openai import OpenAI
from tqdm import tqdm

EMBED_MODEL = "text-embedding-3-small"
client = OpenAI(api_key=st.secrets.get("OPENAI_API_KEY", None))

def build_index(kb_path, index_path, meta_path):
    with open(kb_path, "r", encoding="utf-8") as f:
        kb = json.load(f)

    entries = [entry for entry in kb if entry.get("text")]
    texts = [entry["text"] for entry in entries]

    print(f"🔹 Embedding {len(texts)} chunks from {kb_path}...")
    embeddings = []
    for i in tqdm(range(0, len(texts), 100)):
        batch = texts[i:i+100]
        response = client.embeddings.create(model=EMBED_MODEL, input=batch)
        embeddings.extend([e.embedding for e in response.data])

    embeddings = np.array(embeddings).astype("float32")
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    faiss.write_index(index, index_path)

    records = [
        {
            "id": i,
            "url": entry.get("url", ""),
            "title": entry.get("title", ""),
            "chunk_id": entry.get("chunk_id", 0),
            "text": entry["text"],
        }
        for i, entry in enumerate(entries)
    ]
    meta = {"records": records}
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"✅ Saved {index_path} and {meta_path}")

if __name__ == "__main__":
    build_index("data/haw_kb_en.json", "data/haw_kb_en.index", "data/haw_kb_meta_en.json")
    build_index("data/haw_kb_de.json", "data/haw_kb_de.index", "data/haw_kb_meta_de.json")