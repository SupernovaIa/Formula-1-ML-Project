from fastapi import FastAPI

from backend.routers import clustering, predictions, races, seasons

app = FastAPI(title="Formula 1 ML Project API")

app.include_router(races.router)
app.include_router(seasons.router)
app.include_router(clustering.router)
app.include_router(predictions.router)


@app.get("/health")
def health():
    return {"status": "ok"}
