# Architecture

Three layers, one direction of dependency: `frontend/` calls `backend/`, `backend/`
calls `src/`. Nothing skips a layer.

```
frontend/  (React + Vite)
    │  HTTP (fetch)
    ▼
backend/   (FastAPI — thin wrapper, no business logic of its own)
    │  plain function calls
    ▼
src/       (data processing, ML models, RAG — the actual logic)
```

## `src/` — business logic

Untouched by the backend/frontend rewrite. Everything here is plain Python
functions operating on pandas DataFrames, returning either DataFrames or
Plotly `Figure` objects — written for the original notebooks and reused as-is.

- `dashboard/race.py`, `dashboard/season.py` — race/season data + Plotly figures.
- `circuit_clustering_model/` — K-Means clustering of circuits by technical profile.
- `race_prediction_model/` — feature engineering + classifier training (XGBoost).
- `preprocess.py` — the `Encoding` class (target/ordinal/onehot/frequency) and `scale_df`.
- `rag.py` — chatbot logic (RAG over a small doc corpus in `docs/`, via
  LangChain + Chroma). No Streamlit coupling — `build_vector_db()` builds the
  (single, shared) vector store once, and `stream_rag_response()` takes the
  full conversation history as a plain argument instead of reading
  `st.session_state`.

If you're improving models or features, this is the only place to touch —
`backend/` should never grow business logic of its own.

### `scripts/` — reproducing the data and models

`notebook/`'s `A1`→`A2` (clustering) and `B1`→`B2` (classification) pairs are
real EDA/model-comparison notebooks — they call into `src/`, not the other
way around (see `notebook/README.md`) — but the *order* to run them in, and
which choices from that exploration actually made it into production data,
used to live only in the notebook filenames and a human's memory.
`scripts/build_pipeline.py` codifies that: a staged CLI
(extract → engineer features → preprocess → train) that reproduces
`data/output/*.csv`, `data/preprocessed/circuits_scaled.csv` and
`model/*.pkl` from scratch. Each stage is idempotent (skipped if its output
file already exists) since raw extraction hits FastF1/Ergast over the
network and can be slow — `--force` recomputes only the cheap deterministic
stages, `--force-extract` opts into re-extracting from the network too. The
notebooks remain the right place for anything exploratory (trying other
clustering algorithms, comparing classifiers); the script only encodes the
winning choices already made there.

## `backend/` — REST API

FastAPI app that imports `src/` functions and exposes them over HTTP. Rule of
thumb: a router function fetches/loads data, calls one or two `src/`
functions, and serializes the result. If a router starts doing real
computation, that computation belongs in `src/` instead.

- `main.py` — mounts routers, CORS, and a global `ValueError` → HTTP 422
  handler (for "bad but expected" states like missing upstream data, as
  opposed to a bug, which is a HTTP 500).
- `core/session_cache.py` — `fastf1.Session` objects are expensive to load and
  not JSON-serializable, so they're cached in-process (`lru_cache`) keyed by
  `(year, round, session_type)` rather than round-tripped to the frontend.
- `core/serialization.py` — DataFrame/Plotly-Figure → JSON-safe dict.
- `routers/` — one file per resource (`races`, `seasons`, `clustering`,
  `predictions`, `reference`, `chat`). Most endpoints return whatever `src/`'s
  Plotly functions produce, JSON-encoded as-is (`fig.to_json()` shape) — the
  backend does not know or care that the frontend renders with ECharts, not
  Plotly. `chat` is the exception: it returns a plain streamed text response,
  not JSON — see below.

### The chatbot is stateless by design

`POST /chat` takes the *entire* conversation history in the request body
(like the OpenAI chat API) and streams back plain text — there's no
server-side session, no session ID, nothing keyed per user. This works
because the one thing that would normally need session state — the vector
DB — is the same for every user (one fixed document corpus in `docs/`), so
it's built once lazily on first request (`backend/routers/chat.py`'s
`_get_vector_db()`) and reused for everyone, instead of per-session like the
old Streamlit version did. If the docs ever became per-user (e.g. uploaded
files), this would need to change; for a shared reference corpus, it doesn't.

## `frontend/` — dashboard

React + Vite SPA. Talks to `backend/` exclusively through
`src/api/client.js` — no component calls `fetch` directly.

- `pages/` — one per dashboard section (Home, Race Report, Season Report,
  Circuit Clustering, Winner Prediction, Chatbot).
- `lib/plotlyAdapters.js` — **this is the one place that knows the backend
  still speaks Plotly's JSON shape.** Each function takes a raw Plotly figure
  and returns an ECharts `option` object for one specific trace shape (bar,
  multi-line, scatter groups, radar, track map, telemetry, tyre strategy,
  pace boxplot). Adding a new chart type from the backend means adding one
  function here, not touching `backend/` or `src/`.
- `components/EChart.jsx` — generic renderer for an ECharts `option`, lazy-loaded
  so the ~1MB echarts bundle isn't in every page's initial load.
- `hooks/useAsync.js` — the fetch-on-mount-with-deps pattern every page uses.

### Known simplifications in the adapter layer

- **Pace chart**: the backend's `violin` trace (via Plotly) is approximated as
  an ECharts `boxplot` — ECharts has no native violin series, so the density
  silhouette is dropped in favor of median/quartiles/whiskers, with an
  optional jittered-points overlay to compensate.
- **Tyre strategy**: Plotly's floating bars (`base` + `x` offset) have no
  ECharts equivalent, so it's drawn with a hand-rolled `custom` series
  (`renderItem` + `api.coord`).
- **Chatbot language mirroring**: the system prompt asks the model to answer
  in the question's language, which holds for in-scope answers, but for
  out-of-scope questions its decline message sometimes mirrors the (Spanish)
  source document's language instead. Cosmetic, not a correctness issue — see
  `PLAN.md`.

## Why this shape

- **`src/` staying untouched** meant Streamlit → React and Plotly → ECharts
  were both done without any risk to the actual data/ML logic — the backend
  port and the chart migration were pure refactors from `src/`'s point of
  view.
- **Backend passes through Plotly JSON instead of restructuring it** because
  `src/`'s plotting functions already encode a lot of domain knowledge (team
  colors, corner annotations, axis titles) — reshaping that into a
  frontend-agnostic format would mean re-deriving it in Python, then
  re-deriving it again in the adapters. Translating once, in the adapters, was
  the smaller change.
- **`circuitId` is still ordinal-encoded**, not clustered, despite being a
  known modeling smell — empirically tested and found not to help XGBoost.
  See `PLAN.md` for the experiment and numbers.

## Where else to look

- `PLAN.md` — current priorities, status, and the reasoning behind past
  decisions (what's done, what's deliberately deferred, and why).
- `notebook/README.md` — the original ML methodology (clustering metrics,
  classification model comparison).
- `README.md` — install and run instructions.
