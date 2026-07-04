from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.routers import clustering, predictions, races, reference, seasons

app = FastAPI(title="Formula 1 ML Project API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(ValueError)
def value_error_handler(request: Request, exc: ValueError):
    # Data/model functions raise ValueError for bad-but-expected states
    # (e.g. missing upstream data) — surface it as a clean 422, not a 500.
    return JSONResponse(status_code=422, content={"detail": str(exc)})


app.include_router(races.router)
app.include_router(seasons.router)
app.include_router(clustering.router)
app.include_router(predictions.router)
app.include_router(reference.router)


@app.get("/health")
def health():
    return {"status": "ok"}
