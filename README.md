# Formula 1 Data Analysis and AI Chatbot

## 📜 Project overview

Formula 1 is one of the most competitive and technologically advanced sports in the world. Teams invest millions in collecting, analyzing, and applying detailed data on car performance, driver strategies, and race telemetry. 

However, much of this technical information is not effectively communicated to the general public, sports analysts, or potential sponsors, limiting their ability to understand performance insights and justify investments.

### Objective

The objective of this project is to analyze and effectively communicate the performance of drivers and teams throughout a Formula 1 season, leveraging insights derived from data and machine learning models.

This will help engage fans and enrich sports narratives by providing a clearer understanding of race performances and strategic decisions.

### Exploratory data analysis
- Analysis of general season results, including championship points, podiums, victories, and fastest laps.
- Detailed breakdown per Grand Prix, focusing on:
  - Qualifying and race performance.
  - Tire management and race strategies.
  - Lap-by-lap analysis of position changes.

### Machine learning models

- **Clustering of circuits** based on technical characteristics to identify similar track profiles.
- **Classification model for victory prediction**, using past race data and starting grid positions.

### AI Chatbot

- Built on **LangChain** and **vector databases** (retrieves from transcribed F1-related videos).
- Proof of concept covering the **2024 Australian Grand Prix** only.
- Currently has no UI — its previous Streamlit-based interface was removed along with the rest of
  the app (see below). It's pending a redesign; the logic lives in `src/rag.py` as reference.

## 🏛️ Project Structure

```
Formula-1-ML-Project
├── backend/    # FastAPI app exposing the dashboard/model logic over REST
├── frontend/   # React + Vite dashboard (consumes the backend API)
├── data/       # Folder containing datasets
├── docs/       # Documentation files for RAG
├── model/      # Machine learning models
├── notebook/   # Jupyter Notebooks for EDA and modeling
├── src/        # Data processing, clustering/classification and RAG logic
├── .env        # Environment variables (not tracked, see .env.example)
├── .env.example  # Template for required environment variables
├── .gitignore  # Git ignore file
├── pyproject.toml  # Project metadata and dependencies (uv)
├── README.md   # Project documentation
└── PLAN.md     # Continuation plan and priorities
```

## ⚙ Installation and requirements

This project was developed in Python 3.12 and uses [`uv`](https://docs.astral.sh/uv/)
for dependency management, plus a Node.js frontend. To set it up:

1. Clone the repository:
   ```bash
   git clone https://github.com/SupernovaIa/Formula-1-ML-Project
   cd Formula-1-ML-Project
   ```
2. Install backend dependencies:
   ```bash
   uv sync
   ```
3. Run the notebooks for data extraction, data analysis and machine learning models.
4. Start the backend API:
   ```bash
   uv run uvicorn backend.main:app --reload
   ```
5. In a separate terminal, install and start the frontend:
   ```bash
   cd frontend
   npm install
   npm run dev
   ```
6. Open the dashboard at `http://localhost:5173`.

> **macOS note:** `xgboost` needs the OpenMP runtime, which doesn't ship with the
> Python package. Install it once with `brew install libomp`.

### Required libraries:

- **OpenAI API Key:** Required for chatbot functionality. ([Documentation](https://platform.openai.com/))
- **ChromaDB:** Vector database for storing chatbot-related data ([Documentation](https://github.com/chroma-core/chroma)).
- **Category Encoders:** Encoding categorical variables for ML models ([Documentation](https://contrib.scikit-learn.org/category_encoders/)).

- **Pandas:** Data manipulation ([Documentation](https://pandas.pydata.org/))
- **NumPy:** Numerical data processing ([Documentation](https://numpy.org/))
- **Plotly:** Interactive visualizations ([Documentation](https://plotly.com/))
- **FastF1:** F1 telemetry and race data extraction ([Documentation](https://theoehrly.github.io/Fast-F1/))
- **Scikit-learn:** Machine Learning algorithms ([Documentation](https://scikit-learn.org/))
- **XGBoost:** Advanced ML models ([Documentation](https://xgboost.readthedocs.io/))
- **LangChain:** AI chatbot integration ([Documentation](https://python.langchain.com/))
- **FastAPI:** Backend API framework ([Documentation](https://fastapi.tiangolo.com/))
- **React + Vite:** Frontend dashboard ([Documentation](https://react.dev/))

## 🚀 Next steps

See [`PLAN.md`](./PLAN.md) for the up-to-date priorities and roadmap.

## ✍️ Author

Javier Carreira - Lead Developer  
GitHub: [SupernovaIa](https://github.com/SupernovaIa)