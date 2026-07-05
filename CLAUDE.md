# CLAUDE.md

Formula 1 data analysis, ML models (circuit clustering + race-winner
prediction) and a dashboard. See `ARCHITECTURE.md` for the full picture;
`PLAN.md` for current priorities and status.

## The one rule

Three layers, one direction: `frontend/` → `backend/` → `src/`.

- `src/` is the only place with business logic (data processing, ML, RAG).
- `backend/` (FastAPI) only wraps `src/` functions behind REST endpoints —
  it should never contain computation that belongs in `src/`.
- `frontend/` (React) only talks to `backend/` via `src/api/client.js` — never
  imports from `src/` or reads data files directly.

Adding a chart type: `src/` returns a Plotly figure like everything else, the
backend serializes it as-is, and the ONLY new frontend code is a translator
function in `frontend/src/lib/plotlyAdapters.js`. Don't reshape data in the
backend to suit the frontend's chart library — see `ARCHITECTURE.md` for why.

## Running it locally

```bash
uv sync
uv run uvicorn backend.main:app --reload   # backend on :8000

cd frontend && npm install && npm run dev   # frontend on :5173, separate terminal
```

macOS: `xgboost` needs `brew install libomp` (not a Python package).

## Verifying changes

There's no test suite yet. When you change something:

- **Backend/`src/` logic**: smoke-test with `curl` against the running
  server using real data (a specific year/round that's known to work, e.g.
  `2023`/`1`), not mocks — this codebase has real data-quality gaps (missing
  fields for some sessions, circuits absent from the clustering dataset,
  etc.) that only show up against actual FastF1 responses.
- **Frontend**: `npm run build` (catches import/syntax errors) and
  `npm run lint`. There's no browser automation available in this
  environment — after a UI change, say so explicitly and ask for a visual
  check rather than assuming it renders correctly just because it builds.
- **New chart adapter**: verify it against the actual JSON the corresponding
  endpoint returns (trace `type`, field names) before wiring it into a page —
  the Plotly trace shapes vary a lot (bar vs scatter vs scatterpolar vs
  violin) and guessing the shape has caused real bugs before (e.g. an ECharts
  value axis defaulting to a zero baseline silently flattened a chart with a
  narrow data range).

## Known dormant/incomplete pieces

- The chatbot (`src/rag.py` + `backend/routers/chat.py` +
  `frontend/src/pages/ChatBot.jsx`) needs a real `OPENAI_API_KEY` in `.env` to
  work — without one, `/chat` returns a clean 500, the rest of the app is
  unaffected. It's stateless (see `ARCHITECTURE.md`) and scoped to a single
  document (`docs/AUS_2024_SUM.txt`, the 2024 Australian GP) — adding more
  races is a data task (drop more files in `docs/`), not a code change.
- `circuitId` is ordinal-encoded in the classification model, which is a
  known modeling smell — deliberately left as-is after testing showed
  clustering-based alternatives don't improve XGBoost's metrics (see
  `PLAN.md` for the numbers before re-litigating this).
