"""Core RAG pipeline: load PDF -> chunk -> embed -> store -> retrieve -> answer."""
import hashlib
import io
import math
import re
import time
from typing import Iterator

import chromadb
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

from . import config


# ---------- embeddings ----------
class FakeEmbeddings:
    """Deterministic bag-of-words hashing embeddings. Offline, for tests and demos."""
    DIM = 256

    def _embed(self, text: str) -> list[float]:
        vec = [0.0] * self.DIM
        for tok in re.findall(r"[a-z0-9]+", text.lower()):
            h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
            vec[h % self.DIM] += 1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    def embed_documents(self, texts):
        return [self._embed(t) for t in texts]

    def embed_query(self, text):
        return self._embed(text)


def get_embeddings():
    if config.EMBEDDINGS == "fake":
        return FakeEmbeddings()
    from langchain_openai import OpenAIEmbeddings
    return OpenAIEmbeddings(model=config.EMBED_MODEL)


# ---------- vector store ----------
_client = None


def _collection():
    global _client
    if _client is None:
        _client = chromadb.PersistentClient(path=config.CHROMA_DIR)
    return _client.get_or_create_collection("docuchat", metadata={"hnsw:space": "cosine"})


# ---------- ingestion ----------
def extract_pages(pdf_bytes: bytes) -> list[tuple[int, str]]:
    reader = PdfReader(io.BytesIO(pdf_bytes))
    return [(i + 1, (p.extract_text() or "").strip()) for i, p in enumerate(reader.pages)]


def chunk_pages(pages, chunk_size=None, overlap=None) -> list[dict]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size or config.CHUNK_SIZE,
        chunk_overlap=overlap if overlap is not None else config.CHUNK_OVERLAP,
    )
    chunks = []
    for page_no, text in pages:
        for piece in splitter.split_text(text):
            chunks.append({"text": piece, "page": page_no})
    return chunks


def ingest_pdf(user_id: int, doc_id: int, filename: str, pdf_bytes: bytes) -> int:
    chunks = chunk_pages(extract_pages(pdf_bytes))
    if not chunks:
        return 0
    texts = [c["text"] for c in chunks]
    _collection().add(
        ids=[f"{doc_id}-{i}" for i in range(len(chunks))],
        documents=texts,
        embeddings=get_embeddings().embed_documents(texts),
        metadatas=[
            {"user_id": user_id, "doc_id": doc_id, "filename": filename, "page": c["page"]}
            for c in chunks
        ],
    )
    return len(chunks)


def delete_document(doc_id: int):
    _collection().delete(where={"doc_id": doc_id})


# ---------- retrieval ----------
def retrieve(user_id: int, question: str, top_k=None, doc_id=None) -> list[dict]:
    where = {"user_id": user_id}
    if doc_id:
        where = {"$and": [{"user_id": user_id}, {"doc_id": doc_id}]}
    res = _collection().query(
        query_embeddings=[get_embeddings().embed_query(question)],
        n_results=top_k or config.TOP_K,
        where=where,
    )
    out = []
    for text, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
        out.append({"text": text, "filename": meta["filename"], "page": meta["page"],
                    "score": round(1 - dist, 4)})
    return out


# ---------- generation ----------
SYSTEM_PROMPT = (
    "You answer questions using ONLY the numbered context passages provided. "
    "Cite sources inline like [1], [2]. If the context does not contain the answer, "
    "say you could not find it in the uploaded documents. Do not use outside knowledge."
)


def build_prompt(question: str, passages: list[dict]) -> str:
    ctx = "\n\n".join(
        f"[{i}] ({p['filename']}, page {p['page']})\n{p['text']}" for i, p in enumerate(passages, 1)
    )
    return f"Context:\n{ctx}\n\nQuestion: {question}"


def stream_answer(question: str, passages: list[dict]) -> Iterator[str]:
    if not passages:
        yield "I could not find anything in your uploaded documents. Upload a PDF first."
        return
    if config.LLM == "fake":
        # Offline mode: echo the best passage so the pipeline can be tested end to end.
        yield f"Based on the documents: {passages[0]['text'][:300]} [1]"
        return
    from langchain_openai import ChatOpenAI
    llm = ChatOpenAI(model=config.CHAT_MODEL, temperature=0, streaming=True)
    messages = [("system", SYSTEM_PROMPT), ("human", build_prompt(question, passages))]
    for chunk in llm.stream(messages):
        if chunk.content:
            yield chunk.content


# ---------- response cache ----------
_cache: dict[str, tuple[float, str, list]] = {}


def cache_key(user_id: int, question: str, doc_id) -> str:
    raw = f"{user_id}|{doc_id}|{question.strip().lower()}"
    return hashlib.sha256(raw.encode()).hexdigest()


def cache_get(key: str):
    hit = _cache.get(key)
    if hit and time.time() - hit[0] < config.CACHE_TTL_SECONDS:
        return hit[1], hit[2]
    _cache.pop(key, None)
    return None


def cache_set(key: str, answer: str, sources: list):
    _cache[key] = (time.time(), answer, sources)


def cache_clear():
    _cache.clear()
