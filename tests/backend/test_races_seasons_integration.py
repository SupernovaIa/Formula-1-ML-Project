"""Integration tests against live FastF1/Ergast data.

Excluded from the default `pytest` run (see the `integration` marker in
pyproject.toml) since they need network access, can be slow, and depend on
Ergast's own uptime (observed to intermittently time out independent of which
race is requested) - run explicitly with `pytest -m integration`. 2023/1 is
also CLAUDE.md's own reference race for manual smoke-testing, and happens to
double as a regression test for the plot_results NaN-TeamName bug (see
PLAN.md): it has unclassified results, so /standings for it is expected to be
a 422, not a 200.
"""

import pytest
from fastapi.testclient import TestClient

from backend.main import app

pytestmark = pytest.mark.integration

client = TestClient(app)

YEAR = 2023
ROUND = 1


def test_race_results_returns_classified_drivers():
    response = client.get(f"/races/{YEAR}/{ROUND}/R/results")

    assert response.status_code == 200
    results = response.json()
    assert len(results) > 0


def test_qualifying_results_uses_the_qualifying_branch():
    response = client.get(f"/races/{YEAR}/{ROUND}/Q/results")

    assert response.status_code == 200
    assert len(response.json()) > 0


def test_race_drivers_returns_abbreviation_and_full_name():
    response = client.get(f"/races/{YEAR}/{ROUND}/R/drivers")

    assert response.status_code == 200
    drivers = response.json()
    assert len(drivers) > 0
    assert {"Abbreviation", "FullName"} == set(drivers[0].keys())


def test_standings_plot_on_a_race_with_unclassified_results_is_a_clean_422():
    # 2023 round 1 has zero-classified-result rows with a NaN TeamName (see
    # PLAN.md) - plot_results raises ValueError for this, surfaced as a 422
    # rather than the TypeError crash it used to be.
    response = client.get(f"/races/{YEAR}/{ROUND}/R/standings")

    assert response.status_code == 422


def test_position_changes_plot_returns_a_plotly_figure():
    response = client.get(f"/races/{YEAR}/{ROUND}/R/position-changes")

    assert response.status_code == 200
    assert "data" in response.json()


def test_track_plot_returns_a_plotly_figure():
    response = client.get(f"/races/{YEAR}/{ROUND}/R/track")

    assert response.status_code == 200
    assert "data" in response.json()


def test_drivers_championship_returns_a_plotly_figure():
    response = client.get(f"/seasons/{YEAR}/drivers-championship")

    assert response.status_code == 200
    assert "data" in response.json()


def test_constructors_championship_returns_a_plotly_figure():
    response = client.get(f"/seasons/{YEAR}/constructors-championship")

    assert response.status_code == 200
    assert "data" in response.json()
