import json
from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sqlalchemy.orm import Session

from . import rag
from .auth import create_token, current_user, hash_password, verify_password
from .db import Document, User, get_db, init_db

app = FastAPI(title="DocuChat", description="RAG assistant: upload PDFs, ask questions, get cited answers.")
init_db()

MAX_UPLOAD_BYTES = 20 * 1024 * 1024


class Credentials(BaseModel):
    email: str
    password: str


class ChatRequest(BaseModel):
    question: str
    doc_id: int | None = None
    top_k: int | None = None


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/auth/register", status_code=201)
def register(body: Credentials, db: Session = Depends(get_db)):
    if len(body.password) < 8:
        raise HTTPException(400, "Password must be at least 8 characters")
    if db.query(User).filter_by(email=body.email.lower()).first():
        raise HTTPException(409, "Email already registered")
    user = User(email=body.email.lower(), password_hash=hash_password(body.password))
    db.add(user)
    db.commit()
    return {"id": user.id, "email": user.email}


@app.post("/auth/login")
def login(body: Credentials, db: Session = Depends(get_db)):
    user = db.query(User).filter_by(email=body.email.lower()).first()
    if not user or not verify_password(body.password, user.password_hash):
        raise HTTPException(401, "Wrong email or password")
    return {"access_token": create_token(user.id), "token_type": "bearer"}


@app.post("/documents", status_code=201)
def upload_document(
    file: UploadFile = File(...), user: User = Depends(current_user), db: Session = Depends(get_db)
):
    if not (file.filename or "").lower().endswith(".pdf"):
        raise HTTPException(400, "Only PDF files are supported")
    data = file.file.read()
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "File too large (20 MB max)")
    doc = Document(user_id=user.id, filename=file.filename, num_chunks=0)
    db.add(doc)
    db.flush()
    try:
        doc.num_chunks = rag.ingest_pdf(user.id, doc.id, file.filename, data)
    except Exception as e:
        db.rollback()
        raise HTTPException(422, f"Could not read PDF: {e}")
    if doc.num_chunks == 0:
        db.rollback()
        raise HTTPException(422, "No extractable text found (scanned PDF?)")
    db.commit()
    rag.cache_clear()
    return {"id": doc.id, "filename": doc.filename, "chunks": doc.num_chunks}


@app.get("/documents")
def list_documents(user: User = Depends(current_user), db: Session = Depends(get_db)):
    docs = db.query(Document).filter_by(user_id=user.id).order_by(Document.id.desc()).all()
    return [{"id": d.id, "filename": d.filename, "chunks": d.num_chunks,
             "uploaded": d.created_at.isoformat()} for d in docs]


@app.delete("/documents/{doc_id}", status_code=204)
def delete_document(doc_id: int, user: User = Depends(current_user), db: Session = Depends(get_db)):
    doc = db.get(Document, doc_id)
    if not doc or doc.user_id != user.id:
        raise HTTPException(404, "Document not found")
    rag.delete_document(doc_id)
    db.delete(doc)
    db.commit()
    rag.cache_clear()


@app.post("/chat")
def chat(body: ChatRequest, user: User = Depends(current_user), db: Session = Depends(get_db)):
    """Streams server-sent events: one `sources` event, many `token` events, then `done`."""
    if body.doc_id:
        doc = db.get(Document, body.doc_id)
        if not doc or doc.user_id != user.id:
            raise HTTPException(404, "Document not found")

    key = rag.cache_key(user.id, body.question, body.doc_id)
    cached = rag.cache_get(key)

    def events():
        if cached:
            answer, sources = cached
            yield _sse("sources", sources)
            yield _sse("token", answer)
            yield _sse("done", {"cached": True})
            return
        passages = rag.retrieve(user.id, body.question, body.top_k, body.doc_id)
        sources = [{"n": i, "filename": p["filename"], "page": p["page"],
                    "score": p["score"], "snippet": p["text"][:200]}
                   for i, p in enumerate(passages, 1)]
        yield _sse("sources", sources)
        parts = []
        for tok in rag.stream_answer(body.question, passages):
            parts.append(tok)
            yield _sse("token", tok)
        rag.cache_set(key, "".join(parts), sources)
        yield _sse("done", {"cached": False})

    return StreamingResponse(events(), media_type="text/event-stream")


def _sse(event: str, data) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"
