import joblib
import pandas as pd
from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter(prefix="/predictions", tags=["predictions"])

_encoder = joblib.load("model/encoder.pkl")
_scaler = joblib.load("model/scaler.pkl")
_model = joblib.load("model/best_model.pkl")


class WinnerPredictionRequest(BaseModel):
    driver_id: str
    team_id: str
    circuit_id: str
    grid_position: int
    round_number: int
    mean_previous_grid: float
    mean_previous_position: float
    current_driver_wins: int
    current_driver_podiums: int


class WinnerPredictionResponse(BaseModel):
    predicted_winner: bool
    win_probability: float


@router.post("/winner", response_model=WinnerPredictionResponse)
def predict_winner(payload: WinnerPredictionRequest):
    new_data = pd.DataFrame([{
        "DriverId": payload.driver_id,
        "TeamId": payload.team_id,
        "GridPosition": payload.grid_position,
        "round": payload.round_number,
        "MeanPreviousGrid": payload.mean_previous_grid,
        "MeanPreviousPosition": payload.mean_previous_position,
        "CurrentDriverWins": payload.current_driver_wins,
        "CurrentDriverPodiums": payload.current_driver_podiums,
        "circuitId": payload.circuit_id,
    }])

    df_encoded = _encoder.transform(new_data)
    df_scaled = _scaler.transform(df_encoded)
    df_scaled = pd.DataFrame(df_scaled, columns=df_encoded.columns, index=df_encoded.index)

    prediction = _model.predict(df_scaled)
    win_probability = _model.predict_proba(df_scaled)[:, 1]

    return WinnerPredictionResponse(
        predicted_winner=bool(prediction[0] == 1),
        win_probability=float(win_probability[0]),
    )
