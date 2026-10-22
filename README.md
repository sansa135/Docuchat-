
# DocuChat: RAG Document Assistant

Upload PDFs, ask questions in plain language, and get streamed answers with source citations (file name and page number). Built with Python, FastAPI, LangChain, ChromaDB, PostgreSQL, JWT and Docker.

## How it works
1. **Ingest:** PDF text is extracted page by page, split into overlapping chunks, embedded, and stored in ChromaDB with the user, document, file name and page as metadata.
2. **Retrieve:** The question is embedded and the top-k nearest chunks are fetched, filtered to the logged-in user's documents only.
3. **Answer:** The LLM is told to answer only from the numbered passages and cite them as [1], [2]. If the answer isn't in the context, it must say so.
4. **Stream:** `/chat` streams `sources`, then `token`s, then `done`. Repeated questions are served from a TTL cache.

## Run locally
```bash
git clone https://github.com/sansa135/Docuchat-.git
cd Docuchat-
cp .env.example .env          # add your OPENAI_API_KEY
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
| POST | /auth/register, /auth/login | Create an account, get a JWT |
| POST | /documents | Upload a PDF |
| GET | /documents | List your documents |
| DELETE | /documents/{id} | Delete a document and its vectors |
| POST | /chat | Ask a question, get a streamed cited answer |

## Tune and evaluate
Put questions and the phrase each answer must contain in `eval/questions.json`, then run:
```bash
python eval/run_eval.py sample/sample.pdf eval/questions.json
```
It tries different chunk sizes and top-k values and prints the retrieval hit rate. Set `CHUNK_SIZE`, `CHUNK_OVERLAP` and `TOP_K` in `.env` to apply the best settings.

## Tests
`pytest` runs offline with fake embeddings and a fake LLM, so no API key is needed. It covers auth, upload, cited retrieval, caching and per-user isolation.

## Limitations
- Scanned PDFs are rejected because there is no OCR.
- The cache is in-memory, so it works per process. Use Redis for multiple workers.
