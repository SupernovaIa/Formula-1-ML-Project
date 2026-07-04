import pandas as pd
from fastapi import APIRouter

router = APIRouter(prefix="/reference/seasons/{year}", tags=["reference"])


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
