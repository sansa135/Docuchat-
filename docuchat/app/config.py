import os

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./docuchat.db")
JWT_SECRET = os.getenv("JWT_SECRET", "dev-secret-change-me")
JWT_EXPIRE_MINUTES = int(os.getenv("JWT_EXPIRE_MINUTES", "60"))

LLM = os.getenv("LLM", "openai")                 # openai | fake
CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4o-mini")
EMBEDDINGS = os.getenv("EMBEDDINGS", "openai")   # openai | fake
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "100"))
TOP_K = int(os.getenv("TOP_K", "4"))
CHROMA_DIR = os.getenv("CHROMA_DIR", "./chroma_data")
CACHE_TTL_SECONDS = int(os.getenv("CACHE_TTL_SECONDS", "600"))
