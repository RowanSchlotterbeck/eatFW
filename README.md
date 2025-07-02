# EatFW

A full-stack demo that serves restaurant recommendations for Fort Worth, TX. The repository contains:

- `backend/` – FastAPI + ChromaDB + Ollama backend
- `eatfw/` – Next.js 14 front-end (App Router)

Follow the steps below to run the project locally.

---

## Prerequisites

1. **Python 3.10+** (to run the FastAPI backend)
2. **Node.js 18 LTS+** (to run the Next.js front-end)
3. **[Ollama](https://ollama.com/)** installed locally – used for embeddings & LLM calls
   - Pull the required models once:
     ```bash
     ollama pull llama3
     ollama pull nomic-embed-text
     ```

> Tip: All commands assume you are in the repository root `eatFW/`.

---

## 1. Install dependencies

### Backend (Python)

```bash
python -m venv .venv           # create virtual environment – optional but recommended
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r backend/requirements.txt
```

### Front-end (Node)

```bash
cd eatfw
npm install                    # or yarn / pnpm / bun
cd ..                          # return to root if you wish
```

---

## 2. Prepare the vector database

The backend needs a ChromaDB collection populated with restaurant data.

1. Start Ollama in a separate terminal (required for generating embeddings):
   ```bash
   ollama serve
   ```
2. Run the ingestion script once:
   ```bash
   python backend/ingest.py
   ```
   This will read `backend/resturant-data.json`, embed each entry, and create a persistent ChromaDB database at `backend/chroma_db/`.

---

## 3. Run the development servers

Open **two** terminal windows/tabs.

### Terminal 1 – FastAPI backend

```bash
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

### Terminal 2 – Next.js front-end

```bash
npm run dev --prefix eatfw      # or: cd eatfw && npm run dev
```

Visit `http://localhost:3000` in your browser. The front-end is configured to call the backend at `http://localhost:8000`.

---

## 4. Useful commands

- **Format & lint Python**: `ruff format backend && ruff check backend`
- **Run backend tests** : _tests not yet added_
- **Build production front-end**:
  ```bash
  cd eatfw
  npm run build && npm start
  ```

---

## 5. Troubleshooting

- **Ollama not running** – Most errors during ingestion or inference are caused by Ollama not being started. Ensure `ollama serve` is running.
- **Missing models** – Run `ollama pull llama3` and `ollama pull nomic-embed-text` once.
- **Port already in use** – Adjust the `uvicorn` or `npm run dev` port with `--port` or `-p` flags.

---

## License

MIT © 2024 Rowan Schlotterbeck
