# Formula 1 ML Project — Plan

## Current state

Working pipeline covering EDA, circuit clustering (K-Means, 7 clusters), race-winner
classification (XGBoost, 97.4% accuracy), a React + Vite dashboard on top of a
FastAPI backend, and a RAG chatbot (proof of concept, 2024 Australian GP only)
with a working UI again. Dependency management runs on `uv`.

## Priorities

| # | Item | Priority | Status |
|---|------|----------|--------|
| 1 | Rework clustering & classification models (more features, encode `circuitId` from clustering into the predictive model) | **High** | Partially investigated — see below |
| 2 | Replace the dashboard: new frontend, drop Streamlit | **High** | ✅ Done |
| 3 | SQL database layer (stop depending on live FastF1 calls) | Medium | Not started |
| 4 | Expand prediction to more race variables | Medium | Not started |
| 5 | Chatbot: new pass on scope and architecture (not just "add more docs") | Medium | ✅ Done — see below |
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

**Testing & CI.** ✅ Done — added a `pytest` suite (`tests/`) covering `src/`
business logic (encoding, scaling, feature engineering, classifier
determinism) with small synthetic data, plus `TestClient` tests for the
backend routes that don't need live network (`clustering`, `predictions`,
`reference`, `chat`'s error paths) against the real committed
`data/`/`model/` artifacts — no mocks. Routes that do need live FastF1/Ergast
(`races`, `seasons`) are covered too, but marked `integration` and excluded
by default (`pytest -m integration` to run them) since Ergast itself is
observed to intermittently time out independent of which race is requested —
not something a CI gate should depend on. Added `.github/workflows/ci.yml`:
runs the default suite plus `npm run build`/`npm run lint` on every push/PR
to `main`.

**Reproducible data/model pipeline.** ✅ Done — added `scripts/build_pipeline.py`,
a staged, idempotent CLI that codifies the sequence that used to live only in
the order of the `A1`/`A2`/`B1`/`B2` notebooks (extraction → feature
engineering → preprocessing → training). Each stage skips if its output file
already exists, so running it doesn't silently re-hit FastF1/Ergast; raw
extraction only runs on a missing file or `--force-extract`, while
`--force` alone recomputes just the cheap, deterministic stages (feature
engineering, preprocessing, training). The notebooks stay as-is for EDA and
model comparison — the script only encodes the choices already made there
(7-cluster K-Means features, minmax scaling, final XGBoost). Verified by
running it for real: the clustering stages reproduce the committed
`featured_circuits_complete.csv`/`circuits_scaled.csv` bit-for-bit (floating
point noise aside), and the classification stages reproduce
`featured_results.csv` exactly and train a working model end to end
(confirmed via a live `/predictions/winner` call).

Fixed two related bugs found while building this: `src/preprocess.py`'s
`save_objects` and `src/race_prediction_model/classification.py`'s
`fit_model` both hardcoded `../model/...` paths that only worked when run
from inside `notebook/` — now they use the same cwd-aware pattern already
used in `src/*/extract.py`. Also fixed a `preprocess()` crash when called
without a `target_variable` (the clustering pipeline's exact call pattern) —
`df.drop(columns=None)` raises in the installed pandas version instead of
being a no-op.

**Known caveat:** a full retrain via the script's `classification-train`
stage doesn't reproduce the exact same hyperparameters as the currently
committed `model/best_model.pkl` — `ClassificationModels`'s `XGBClassifier()`
has no fixed `random_state`, so its own internal randomness (not the
train/test split, which is seeded) makes `GridSearchCV` pick slightly
different "best" hyperparameters across runs even on identical data.
Pre-existing behavior, not something this change introduced; worth a
`random_state` fix later if bit-for-bit model reproducibility ever matters.

**Trade-off from dropping Streamlit (resolved):** the chatbot (`src/rag.py`) only
ever had a Streamlit UI (`app.py`, now removed) and was tightly coupled to
`st.session_state`. It wasn't ported at the time — that rework is now done, see
below.

**Chatbot rework (item 5), done:** rewrote `src/rag.py` to drop all Streamlit
coupling. Key simplification: the vector DB is the same for every user (one
fixed document corpus), so there's no need for per-session state at all — it's
built once (lazily, on first request) and reused, and conversation history is
stateless (the frontend sends the full message array each turn, like the
OpenAI chat API, instead of a server-side session). Added `backend/routers/chat.py`
(`POST /chat`, streaming response, backend's own `OPENAI_API_KEY`) and a new
`frontend/src/pages/ChatBot.jsx`. Verified end-to-end with a real API key:
correct answers grounded in the doc, correctly declines out-of-scope questions
instead of hallucinating, follow-up questions resolve via conversation history.
Scope unchanged — still just the 2024 Australian GP doc; adding more races is
now purely a data task (drop more `.txt`/`.md` files in `docs/`), no code changes
needed.

**Known rough edge:** the system prompt instructs the model to reply in the
question's language, which works reliably for in-scope answers, but for
out-of-scope questions the model's "I don't have information on that" reply
sometimes mirrors the context document's language (Spanish) instead of the
question's language, even after two rounds of strengthening the prompt
instruction. Cosmetic only — it still correctly declines to fabricate an
answer — not worth chasing further for a single-document proof of concept.

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

- Whether backend + frontend live in this repo or get split into separate repos.

## Suggested phased order

1. ~~**Foundation** — `uv` migration + modularize `src/` into a backend package~~ ✅ Done.
2. ~~**High priority (frontend)** — stand up the new frontend consuming the backend API~~ ✅ Done.
3. **High priority (models)** — ~~fix `circuitId` encoding~~ investigated, no
   change shipped (see above); ~~fix the `plot_results` NaN bug~~ done. Still
   open: add more features / feature selection to clustering & classification.
4. **Medium priority** — SQL layer, expanded prediction targets;
   ~~chatbot rework~~ ✅ done (see above).
5. **Low priority** — refresh data with 2025/2026 seasons once the above is
   stable (avoids re-doing model/feature work twice).
