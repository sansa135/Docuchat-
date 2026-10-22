import os, tempfile
_tmp = tempfile.mkdtemp()
os.environ.update({
    "DATABASE_URL": f"sqlite:///{_tmp}/test.db",
    "CHROMA_DIR": f"{_tmp}/chroma",
    "EMBEDDINGS": "fake",
    "LLM": "fake",
    "JWT_SECRET": "test-secret",
})
