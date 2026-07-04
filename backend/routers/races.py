from fastapi import APIRouter, Query

from backend.core.serialization import df_to_records, fig_to_json
from backend.core.session_cache import get_session
from src.dashboard.race import (
    draw_track,
    get_qualy_results,
    get_race_results,
    plot_driver_pace,
    plot_drivers_pace,
    plot_fastest_laps,
    plot_position_changes,
    plot_results,
    plot_telemetry,
    plot_tyre_strat,
)

router = APIRouter(prefix="/races/{year}/{round_number}/{session_type}", tags=["races"])


def _is_qualifying(session_type: str) -> bool:
    return session_type.lower().startswith("q")


@router.get("/results")
def results(year: int, round_number: int, session_type: str):
    session = get_session(year, round_number, session_type)
    if _is_qualifying(session_type):
        df = get_qualy_results(session)
    else:
        df = get_race_results(session).reset_index(drop=True)
    return df_to_records(df)


@router.get("/drivers")
def drivers(year: int, round_number: int, session_type: str):
    session = get_session(year, round_number, session_type)
    df = session.results[["Abbreviation", "FullName"]].reset_index(drop=True)
    return df_to_records(df)


@router.get("/fastest-laps")
def fastest_laps(year: int, round_number: int, session_type: str):
    session = get_session(year, round_number, session_type)
    return fig_to_json(plot_fastest_laps(session))


@router.get("/track")
def track(year: int, round_number: int, session_type: str):
    session = get_session(year, round_number, session_type)
    return fig_to_json(draw_track(session))


@router.get("/telemetry")
def telemetry(
    year: int,
    round_number: int,
    session_type: str,
    mode: str = "Speed",
    drivers: list[str] = Query(default=[]),
):
    session = get_session(year, round_number, session_type)
    return fig_to_json(plot_telemetry(session, mode=mode, drivers=drivers))


@router.get("/standings")
def standings(year: int, round_number: int, session_type: str):
    session = get_session(year, round_number, session_type)
    return fig_to_json(plot_results(session))


@router.get("/position-changes")
def position_changes(year: int, round_number: int, session_type: str):
    session = get_session(year, round_number, session_type)
    return fig_to_json(plot_position_changes(session))


@router.get("/driver-pace")
def driver_pace(year: int, round_number: int, session_type: str, driver: str, threshold: float = 1.07):
    session = get_session(year, round_number, session_type)
    return fig_to_json(plot_driver_pace(session, driver, threshold))


@router.get("/pace")
def pace(
    year: int,
    round_number: int,
    session_type: str,
    kind: str = "driver",
    threshold: float = 1.07,
    box: bool = False,
):
    session = get_session(year, round_number, session_type)
    return fig_to_json(plot_drivers_pace(session, kind, threshold, box))


@router.get("/tyre-strategy")
def tyre_strategy(year: int, round_number: int, session_type: str):
    session = get_session(year, round_number, session_type)
    return fig_to_json(plot_tyre_strat(session))
