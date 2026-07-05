# Formula 1 ML Project

Formula 1 data analysis and machine learning: circuit clustering, race-winner
prediction, and a dashboard built on top of FastF1 data.

- **EDA & clustering** — circuits grouped by technical profile (K-Means).
- **Race-winner prediction** — XGBoost classifier from grid position and form.
- **Dashboard** — React frontend, FastAPI backend, both consuming `src/`.
- **AI chatbot** — RAG proof of concept over race documents (needs an
  `OPENAI_API_KEY` in `.env`).

For how it's put together, see [`ARCHITECTURE.md`](./ARCHITECTURE.md). For
current priorities and status, see [`PLAN.md`](./PLAN.md).

## Quick start

```bash
git clone https://github.com/SupernovaIa/Formula-1-ML-Project
cd Formula-1-ML-Project

uv sync
uv run uvicorn backend.main:app --reload   # backend on :8000

cd frontend && npm install && npm run dev   # frontend on :5173, separate terminal
```

Open `http://localhost:5173`.

> **macOS:** `xgboost` needs the OpenMP runtime: `brew install libomp`.

Everything works without any further setup except the chatbot, which needs an
OpenAI API key: `cp .env.example .env` and fill in `OPENAI_API_KEY`.

The data and models the dashboard reads (`data/output/`, `data/preprocessed/`,
`model/`) are checked in, so none of the above regenerates them. To rebuild
them from scratch (or after a feature-engineering change), see
`scripts/build_pipeline.py`.

Notebooks for data extraction, EDA and model training live in `notebook/`
(see `notebook/README.md` for methodology).

## Testing

```bash
uv run pytest                # src/ + backend unit/API tests, no network needed
uv run pytest -m integration # also exercises live FastF1/Ergast-backed routes
```

CI (GitHub Actions) runs the default suite plus `npm run build`/`npm run lint`
on every push/PR to `main`.

## Author

Javier Carreira - Lead Developer
GitHub: [SupernovaIa](https://github.com/SupernovaIa)
