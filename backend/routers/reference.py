import pandas as pd
from fastapi import APIRouter, HTTPException

from src.race_prediction_model.feature_engineering import add_features_to_results, team_id_replacement

router = APIRouter(prefix="/reference/seasons/{year}", tags=["reference"])


def _build_engineered_results():
    # Same computation scripts/build_pipeline.py uses to train the
    # classifier, minus dropping `season` - kept here so a specific
    # year+round can be looked up.
    df = pd.read_csv("data/output/results.csv")
    df.dropna(subset=["GridPosition"], inplace=True)
    team_id_replacement(df)
    add_features_to_results(df)
    return df


_engineered_results = _build_engineered_results()


@router.get("/rounds")
def rounds(year: int):
    races = pd.read_csv("data/output/races.csv")
    season_races = races[races["season"] == year].sort_values("round")
    return [
        {"round": int(row["round"]), "circuit_id": row["circuitId"]}
        for _, row in season_races.iterrows()
    ]


@router.get("/rounds/{round_number}/entrants")
def entrants(year: int, round_number: int):
    races = pd.read_csv("data/output/races.csv")
    results = pd.read_csv("data/output/results.csv")

    circuit_row = races[(races["season"] == year) & (races["round"] == round_number)]
    circuit_id = circuit_row["circuitId"].iloc[0] if not circuit_row.empty else None

    mask = (results["season"] == year) & (results["round"] == round_number)
    subset = results[mask]

    return {
        "circuit_id": circuit_id,
        "drivers": sorted(subset["DriverId"].dropna().unique().tolist()),
        "teams": sorted(subset["TeamId"].dropna().unique().tolist()),
    }


@router.get("/rounds/{round_number}/drivers/{driver_id}/form")
def driver_form(year: int, round_number: int, driver_id: str):
    """A driver's real historical form going into a specific race - the same
    features the classifier trains on, computed from history instead of
    typed in by hand."""
    mask = (
        (_engineered_results["season"] == year)
        & (_engineered_results["round"] == round_number)
        & (_engineered_results["DriverId"] == driver_id)
    )
    rows = _engineered_results[mask]
    if rows.empty:
        raise HTTPException(
            status_code=404,
            detail=f"No result found for driver '{driver_id}' in {year} round {round_number}.",
        )

    row = rows.iloc[0]
    return {
        "team_id": row["TeamId"],
        "grid_position": int(row["GridPosition"]),
        "mean_previous_grid": float(row["MeanPreviousGrid"]),
        "mean_previous_position": (
            float(row["MeanPreviousPosition"]) if pd.notna(row["MeanPreviousPosition"]) else None
        ),
        "current_driver_wins": int(row["CurrentDriverWins"]),
        "current_driver_podiums": int(row["CurrentDriverPodiums"]),
        "actual_position": int(row["Position"]) if pd.notna(row["Position"]) else None,
        "actual_winner": bool(row["Winner"]),
    }
