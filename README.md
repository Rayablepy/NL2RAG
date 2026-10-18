# NL2RAG

A local RAG chat app. Upload documents, ask questions in plain English, get answers based on those documents. Everything runs on your own machine from the chat model and the embeddings to the reranker, and the relevance grader. No API keys, no hosted services.

## How it works

Four local models, wired into a LangGraph agent loop:
| Role | Function |
| --- | --- |
| Chat | Served over an OpenAI-compatible API, the main model which answers your questions. |
| Embeddings | Turns document chunks into vectors, stored in Chroma on disk. |
| Reranker | Cross-encoder that narrows 20 retrieved chunks down to the best 3. |
| Grader | Local classifier(laya) that decides whether the retrieved context is relevant enough to answer from. |

### The agent loop

The agent is a LangGraph `StateGraph`. It decides for itself whether it can answer from what it has, and when to try again with a different question.

```
                       ┌────────────────────┐
                       │  query_or_respond  │
                       └─────────┬──────────┘
                          tool call?
                          yes│    │no
                             │    ▼
                       ┌─────▼─────┐  END
                       │  retrieve │
                       └─────┬─────┘
                       ┌─────▼─────┐
                       │ grade_docs│
                       └─────┬─────┘
       ┌─────────────────────┼─────────────────────┐
   below threshold,      below threshold,      above threshold
   steps left          budget exhausted             │
       │                     │                     ▼
┌──────▼──────┐              ▼            ┌─────────▼─────────┐
│rewrite_query│        cannot_answer      │  generate_answer  │
└──────┬──────┘              │            └─────────┬─────────┘
       │                     │                      │
       └──▶ back to start   └────── END ◀───────────┘
```

- **`query_or_respond`** — the model either answers directly or calls the retrieval tool.
- **`retrieve`** — runs the tool, which queries Chroma for 20 candidates and reranks them to 3.
- **`grade_docs`** — Laya scores how relevant that context is.
- Below the threshold, the agent **rewinds and rewrites the question** with improved semantic intent, then starts over. This is the "2-step" part: retrieval and generation are graded separately, so a weak retrieval gets fixed instead of quietly producing a bad answer.
- It gives up once fewer than `REWRITE_BUFFER` steps remain in the budget, returning *"I couldn't find relevant information in the available documents to answer that"* rather than guessing.
- `recursion_limit` caps the total steps so a confused agent can't loop forever.

## Requirements

- Roughly **12 GB of disk** for model weights on a first run
- A GPU is strongly recommended. On Apple Silicon the chat model uses MPS.

## Setup

The project uses [uv](https://docs.astral.sh/uv/) to manage Python and dependencies.

```bash
git clone https://github.com/Rayablepy/NL2SQL.git
cd NL2SQL
uv sync
```

## Configure

Create a `.env` next to `pyproject.toml`. `.env.example` lists every variable.

```dotenv
CHAT_MODEL_NAME=Qwen/Qwen3-4B-Instruct-2507
EMBEDDING_MODEL_NAME=BAAI/bge-base-en-v1.5
RERANKER_MODEL_NAME=BAAI/bge-reranker-v2-m3

# The chat model is served here; the app connects to it.
BASE_URL=http://localhost:8000/v1

# Must stay under the embedding model's max sequence length.
EMBEDDING_MODEL_CONTEXT=510
EMBEDDING_MODEL_CHUNK=150
```

Anything on the Hugging Face Hub works, as long as the chat model ships a chat template with tool-call support. `config.py` also holds the tuning constants:

| Constant | Default | Meaning |
| --- | --- | --- |
| `RELEVANCE_THRESHOLD` | `0.55` | Below this, the grader rejects the context. |
| `REWRITE_BUFFER` | `3` | Step budget left before the agent gives up. |
| `MAX_STEPS` | `15` | Agent recursion limit. |
| `PLANNER_MAX_TOKENS` | `4096` | Chat history passed to the planner. |

## Run

One command:

```bash
uv run python start.py
```

This starts `transformers serve` for the chat model, waits until it reports healthy, then launches the Gradio UI. Gradio prints the URL on startup — by default `http://localhost:7860`.

Ctrl-C shuts down the UI and tears down the model server with it.

`start.py` is a thin Click wrapper, so you can override any of it:

```bash
uv run python start.py --help

uv run python start.py --model Qwen/Qwen3-4B-Thinking-2507 --port 9000
READY_TIMEOUT=3600 uv run python start.py     # slow first download?
```

### Running the pieces by hand

If you'd rather run them separately, that's two terminals.

Terminal 1 — the model server:

```bash
uv run transformers serve Qwen/Qwen3-4B-Instruct-2507 --port 8000
```

Terminal 2 — the UI:

```bash
uv run python gui.py
```

The port doesn't open until the model has finished loading, so it can look hung for a minute or two on a cold start.

## Using the app

- **Upload** files in the left column (`.pdf`, `.csv`, `.txt`, `.docx`, `.xlsx`). They're parsed, chunked, embedded, and written to the Chroma collection at `./chroma_NL2RAG`.
- **Delete** files by ticking them in *Saved Files* and hitting *Delete Selected*. Re-uploading a file replaces its existing chunks.
- **Ask** questions in the chat. The agent decides on its own when it needs to search your documents, hence you don't have to manually tell it to.

## Project layout

```
start.py    Launcher: model server → health check → UI
gui.py      Gradio interface — upload, delete, chat
main.py     LangGraph agent loop: retrieve, grade, rewrite, answer
loader.py   Chunking, Chroma, embeddings, reranker
config.py   Env vars and tuning constants
```

`chroma_NL2RAG/` holds the vector index, `user_data/` holds copies of uploaded files, and `NL2RAG_models/` caches the embedding and reranker weights.

Only `NL2RAG_models/` is gitignored. `chroma_NL2RAG/` and `user_data/` are not, so a careless `git add .` will stage your vector index and your uploaded documents. Add them to `.gitignore` if that's not what you want.

## Notes and current limitations

- **First run is slow.** All four models download on demand, totalling roughly 12 GB. The embedding and reranker weights land in `NL2RAG_models/`; the chat model and grader use your default Hugging Face cache. Subsequent runs are instant.
- **One conversation at a time.** `gui.py` sends every request with the same hardcoded `thread_id`, so all browser sessions share one conversation thread and one in-memory checkpoint. Change that if you want independent sessions.
- **The server is serialised.** `transformers serve` handles one request at a time by design. Fine for local use; not a deployment setup.
- **`reasoning_content` is dropped.** If you switch to a thinking model, its reasoning trace is discarded on the way back through `ChatOpenAI`, hence you'll get the final answer only.
- The agent loads Laya lazily on its first graded retrieval, so that model appears only after you've asked a question against your documents.
