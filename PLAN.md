# Formula 1 ML Project — Plan

## Current state

Working pipeline covering EDA, circuit clustering (K-Means, 7 clusters), race-winner
classification (XGBoost, 97.4% accuracy), a React + Vite dashboard on top of a
FastAPI backend, and a RAG chatbot (proof of concept, 2024 Australian GP only,
currently **without a UI** — see below). Dependency management runs on `uv`.

## Priorities

| # | Item | Priority | Status |
|---|------|----------|--------|
| 1 | Rework clustering & classification models (more features, encode `circuitId` from clustering into the predictive model) | **High** | Not started |
| 2 | Replace the dashboard: new frontend, drop Streamlit | **High** | ✅ Done |
| 3 | SQL database layer (stop depending on live FastF1 calls) | Medium | Not started |
| 4 | Expand prediction to more race variables | Medium | Not started |
| 5 | Chatbot: new pass on scope and architecture (not just "add more docs") | Medium | Not started — no UI right now |
| 6 | Refresh data with 2025 / 2026 seasons | Low | Not started |

## Cross-cutting architecture work

- **Modularize into a real backend.** ✅ Done — `backend/` (FastAPI) wraps the
  existing `src/` logic behind REST endpoints for races, seasons, clustering and
  predictions. `src/` itself wasn't rewritten.
- **Migrate to `uv`.** ✅ Done.
- **Drop Streamlit for the frontend.** ✅ Done — `frontend/` (React + Vite) covers
  the 4 former Streamlit pages (Race Report, Season Report, Circuit Clustering,
  Winner Prediction), talking to the backend over HTTP. `streamlit` is no longer
  a dependency.

**Trade-off from dropping Streamlit:** the chatbot (`src/rag.py`) only ever had a
Streamlit UI (`app.py`, now removed) and is tightly coupled to
`st.session_state`. It wasn't ported — it needs the scope/architecture rework
from item 5 regardless, so porting it as-is would've been throwaway work. The
logic stays in `src/rag.py` as a reference for that rework; there's no working
chatbot UI until then.

**Known pre-existing bug surfaced during the backend port:** `/races/.../standings`
(→ `src/dashboard/race.py::plot_results`) crashes with a `TypeError` when a lap's
`TeamName` is `NaN` for some races (reproduced on 2023 round 1). Pre-existing in
`src/`, not introduced by the backend wrapper — worth fixing as part of item 1.

## Open decisions

- Chatbot: new scope — how many GPs/documents, same RAG approach or revisit it.
- Whether backend + frontend live in this repo or get split into separate repos.

## Suggested phased order

1. ~~**Foundation** — `uv` migration + modularize `src/` into a backend package~~ ✅ Done.
2. ~~**High priority (frontend)** — stand up the new frontend consuming the backend API~~ ✅ Done.
3. **High priority (models)** — rebuild clustering/classification (fix `circuitId`
   encoding, add features); also fix the `plot_results` NaN bug above while in there.
4. **Medium priority** — SQL layer, expanded prediction targets, chatbot
   rework.
5. **Low priority** — refresh data with 2025/2026 seasons once the above is
   stable (avoids re-doing model/feature work twice).
