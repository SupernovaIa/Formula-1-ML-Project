# Formula 1 ML Project — Plan

## Current state

Working pipeline covering EDA, circuit clustering (K-Means, 7 clusters), race-winner
classification (XGBoost, 97.4% accuracy), a Streamlit dashboard, and a RAG chatbot
(proof of concept, 2024 Australian GP only). Solid methodology, but built as a
collection of scripts/notebooks rather than a maintainable app — no backend, no
package manager beyond pinned `requirements.txt`, frontend and business logic
coupled together in Streamlit.

## Priorities

| # | Item | Priority |
|---|------|----------|
| 1 | Rework clustering & classification models (more features, encode `circuitId` from clustering into the predictive model) | **High** |
| 2 | Replace the dashboard: new frontend, drop Streamlit | **High** |
| 3 | SQL database layer (stop depending on live FastF1 calls) | Medium |
| 4 | Expand prediction to more race variables | Medium |
| 5 | Chatbot: new pass on scope and architecture (not just "add more docs") | Medium |
| 6 | Refresh data with 2025 / 2026 seasons | Low |

## Cross-cutting architecture work

These aren't features — they're prerequisites for doing the high-priority items
without rebuilding twice:

- **Modularize into a real backend.** Pull the model/data logic out of `src/` +
  notebook-driven scripts into a proper service layer with a clean API, instead of
  Streamlit calling Python functions directly.
- **Migrate to `uv`** for dependency management, replacing the pip-freeze
  `requirements.txt`.
- **Drop Streamlit for the frontend.** Framework TBD — decouple frontend from
  backend so the dashboard becomes a real client of an API, not an in-process
  script.

## Open decisions

- Backend framework (candidate: FastAPI — reuses what's already built in
  `fastapi-fundamentals` / `hexagonal-architecture`).
- Frontend framework/stack to replace Streamlit.
- Chatbot: new scope — how many GPs/documents, same RAG approach or revisit it.
- Whether backend + frontend live in this repo or get split into separate repos.

## Suggested phased order

1. **Foundation** — `uv` migration + modularize `src/` into a backend package
   with a clean API boundary. No user-visible change yet, but unblocks
   everything below.
2. **High priority** — rebuild clustering/classification on top of the new
   backend (fix `circuitId` encoding, add features); stand up the new frontend
   consuming the backend API instead of Streamlit.
3. **Medium priority** — SQL layer, expanded prediction targets, chatbot
   rework.
4. **Low priority** — refresh data with 2025/2026 seasons once the above is
   stable (avoids re-doing model/feature work twice).
