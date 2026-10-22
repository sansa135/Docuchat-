# DocuChat: RAG Document Assistant

Upload PDFs, ask questions in plain language, get streamed answers with source citations
(file name and page number). Built with Python, FastAPI, LangChain, ChromaDB, PostgreSQL, JWT and Docker.

## How it works
1. **Ingest:** PDF text is extracted per page, split into overlapping chunks (LangChain splitter), embedded, and stored in ChromaDB with `user_id`, `doc_id`, `filename`, `page` metadata.
2. **Retrieve:** the question is embedded and the top-k nearest chunks are fetched, filtered to the logged-in user's documents only.
3. **Answer:** the LLM is told to answer only from the numbered passages and cite them as [1], [2]. If the answer isn't in the context it must say so.
4. **Stream:** `/chat` streams server-sent events: `sources`, then `token`s, then `done`. Repeated questions are served from a TTL cache.

## Run locally
```bash
cp .env.example .env          # add OPENAI_API_KEY
pip install -r requirements.txt
uvicorn app.main:app --reload
```
Swagger docs: http://localhost:8000/docs

## Run with Docker (PostgreSQL)
```bash
cp .env.example .env
docker compose up --build
```

## Try it
```bash
curl -X POST localhost:8000/auth/register -H 'content-type: application/json' -d '{"email":"me@x.com","password":"password123"}'
TOKEN=$(curl -s -X POST localhost:8000/auth/login -H 'content-type: application/json' -d '{"email":"me@x.com","password":"password123"}' | python -c 'import sys,json;print(json.load(sys.stdin)["access_token"])')
curl -X POST localhost:8000/documents -H "Authorization: Bearer $TOKEN" -F file=@sample/sample.pdf
curl -N -X POST localhost:8000/chat -H "Authorization: Bearer $TOKEN" -H 'content-type: application/json' -d '{"question":"How many days to return a purchase?"}'
```

## API
| Method | Path | Purpose |
|---|---|---|
| POST | /auth/register, /auth/login | create account, get JWT |
| POST | /documents | upload a PDF (auth) |
| GET | /documents | list your documents |
| DELETE | /documents/{id} | delete a document and its vectors |
| POST | /chat | ask a question, streamed cited answer |

## Tune and evaluate
Edit `eval/questions.json` with questions and the phrase each answer must contain, then:
```bash
python eval/run_eval.py sample/sample.pdf eval/questions.json
```
It sweeps chunk size and top-k and prints the retrieval hit rate, so you can pick settings with data.
Set `CHUNK_SIZE`, `CHUNK_OVERLAP`, `TOP_K` in `.env` to apply them.

## Tests (offline)
`pytest` runs with fake embeddings and a fake LLM (no API key needed). It covers auth, upload, cited retrieval, caching, and per-user isolation.

## Limitations
- Scanned PDFs (images) are rejected; there is no OCR.
- The cache is in-memory, so it is per process; use Redis for multiple workers.
- Answer quality with a real LLM is untested here; run the eval with your own documents.
