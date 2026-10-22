"""Retrieval evaluation: sweep chunk size and top-k, report hit rate.

A question is a "hit" if any retrieved chunk contains the expected phrase.
Usage:
    python eval/run_eval.py sample/sample.pdf eval/questions.json
Set EMBEDDINGS=fake to run offline; otherwise OpenAI embeddings are used.
"""
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("CHROMA_DIR", tempfile.mkdtemp())

import chromadb  # noqa: E402
from app import rag  # noqa: E402


def evaluate(pdf_path, questions, chunk_size, overlap, top_k):
    pages = rag.extract_pages(open(pdf_path, "rb").read())
    chunks = rag.chunk_pages(pages, chunk_size, overlap)
    emb = rag.get_embeddings()
    client = chromadb.EphemeralClient()
    name = f"eval_{chunk_size}_{top_k}_{os.getpid()}"
    col = client.get_or_create_collection(name, metadata={"hnsw:space": "cosine"})
    texts = [c["text"] for c in chunks]
    col.add(ids=[str(i) for i in range(len(texts))], documents=texts,
            embeddings=emb.embed_documents(texts))
    hits = 0
    for q in questions:
        res = col.query(query_embeddings=[emb.embed_query(q["question"])],
                        n_results=min(top_k, len(texts)))
        if any(q["expected_phrase"].lower() in d.lower() for d in res["documents"][0]):
            hits += 1
    client.delete_collection(name)
    return hits / len(questions), len(chunks)


if __name__ == "__main__":
    pdf, qfile = sys.argv[1], sys.argv[2]
    questions = json.load(open(qfile))
    print(f"{'chunk':>6} {'overlap':>8} {'top_k':>6} {'chunks':>7} {'hit_rate':>9}")
    for size in (200, 400, 800):
        for k in (1, 2, 4):
            rate, n = evaluate(pdf, questions, size, size // 8, k)
            print(f"{size:>6} {size//8:>8} {k:>6} {n:>7} {rate:>9.0%}")
