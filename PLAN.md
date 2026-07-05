# Formula 1 ML Project — Plan

## Current state

Working pipeline covering EDA, circuit clustering (K-Means, 7 clusters), race-winner
classification (XGBoost, 97.4% accuracy), a React + Vite dashboard on top of a
FastAPI backend, and a RAG chatbot (proof of concept, 2024 Australian GP only,
currently **without a UI** — see below). Dependency management runs on `uv`.

## Priorities

| # | Item | Priority | Status |
|---|------|----------|--------|
| 1 | Rework clustering & classification models (more features, encode `circuitId` from clustering into the predictive model) | **High** | Partially investigated — see below |
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

**Bug fixed during the backend port:** `/races/.../standings`
(→ `src/dashboard/race.py::plot_results`) used to crash with a `TypeError` when a
lap's `TeamName` was `NaN` (some races have zero classified results with a
recorded time delta, e.g. 2023 round 1). Now raises a clear `ValueError` instead,
surfaced by the backend as a 422.

**Investigated: encoding `circuitId` via circuit clustering instead of the
current `OrdinalEncoder`.** Tested 5 variants (ordinal, cluster one-hot,
circuitId target-encoded, target-encoded+cluster combined) on identical
train/test splits with XGBoost. Result: **clustering does not improve the
model** — the current ordinal encoding matches or beats every alternative on
accuracy/precision/recall/F1/AUC. Likely reason: tree-based models aren't hurt
by an ordinal encoding's implied (false) ordering the way linear models would
be, so the "conceptual bug" doesn't cost anything in practice here, while
collapsing 30 circuits into 7 clusters throws away real per-circuit signal.
Also found: 434 rows (~7%) have no clustering data at all (5 circuits dropped
from the calendar pre-2018, out of the app's 2018–2024 range, plus one 2020
one-off). **Decision: kept the existing ordinal encoding, no model change
shipped.** Worth revisiting if the model ever needs to generalize to circuits
with little/no race history, since that's the scenario clustering would
actually help with and this test didn't cover it.

## Open decisions

- Chatbot: new scope — how many GPs/documents, same RAG approach or revisit it.
- Whether backend + frontend live in this repo or get split into separate repos.

## Suggested phased order

1. ~~**Foundation** — `uv` migration + modularize `src/` into a backend package~~ ✅ Done.
2. ~~**High priority (frontend)** — stand up the new frontend consuming the backend API~~ ✅ Done.
3. **High priority (models)** — ~~fix `circuitId` encoding~~ investigated, no
   change shipped (see above); ~~fix the `plot_results` NaN bug~~ done. Still
   open: add more features / feature selection to clustering & classification.
4. **Medium priority** — SQL layer, expanded prediction targets, chatbot
   rework.
5. **Low priority** — refresh data with 2025/2026 seasons once the above is
   stable (avoids re-doing model/feature work twice).
