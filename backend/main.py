from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.routers import clustering, predictions, races, reference, seasons

app = FastAPI(title="Formula 1 ML Project API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(races.router)
app.include_router(seasons.router)
app.include_router(clustering.router)
app.include_router(predictions.router)
app.include_router(reference.router)


@app.get("/health")
def health():
    return {"status": "ok"}
