import pandas as pd
from fastapi import APIRouter

from backend.core.serialization import fig_to_json
from backend.core.session_cache import get_session
from src.dashboard.season import (
    get_constructor_championship,
    get_drivers_championship,
    get_results,
    plot_constructors_championship,
    plot_drivers_championship,
)

router = APIRouter(prefix="/seasons/{year}", tags=["seasons"])


def _load_season_df(year: int) -> pd.DataFrame:
    try:
        return pd.read_csv(f"data/seasons/{year}.csv", index_col=0)
    except FileNotFoundError:
        return get_results(year)


@router.get("/drivers-championship")
def drivers_championship(year: int, top: int | None = 10):
    df_season = _load_season_df(year)
    df_drivers = get_drivers_championship(df_season)
    session = get_session(year, 1, "R")
    return fig_to_json(plot_drivers_championship(df_drivers, session, top=top))


@router.get("/constructors-championship")
def constructors_championship(year: int, top: int | None = None):
    df_season = _load_season_df(year)
    df_constructors = get_constructor_championship(df_season)
    session = get_session(year, 1, "R")
    return fig_to_json(plot_constructors_championship(df_constructors, session, top=top))
