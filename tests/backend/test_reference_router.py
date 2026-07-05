from fastapi.testclient import TestClient

from backend.main import app

client = TestClient(app)


def test_rounds_returns_every_round_of_a_known_season():
    response = client.get("/reference/seasons/2023/rounds")

    assert response.status_code == 200
    rounds = response.json()
    assert len(rounds) > 0
    assert all({"round", "circuit_id"} == set(r.keys()) for r in rounds)
    assert [r["round"] for r in rounds] == sorted(r["round"] for r in rounds)


def test_rounds_unknown_season_returns_empty_list():
    response = client.get("/reference/seasons/1950/rounds")

    assert response.status_code == 200
    assert response.json() == []


def test_entrants_returns_drivers_and_teams_for_a_known_race():
    response = client.get("/reference/seasons/2023/rounds/1/entrants")

    assert response.status_code == 200
    body = response.json()
    assert body["circuit_id"] is not None
    assert len(body["drivers"]) > 0
    assert len(body["teams"]) > 0
    assert body["drivers"] == sorted(body["drivers"])


def test_entrants_unknown_round_returns_none_circuit_and_empty_lists():
    response = client.get("/reference/seasons/2023/rounds/999/entrants")

    assert response.status_code == 200
    body = response.json()
    assert body["circuit_id"] is None
    assert body["drivers"] == []
    assert body["teams"] == []


def test_driver_form_returns_engineered_features_for_a_known_race():
    # 2023 round 1: Verstappen started and won from pole, fresh season so
    # current wins/podiums reset to 0 - a good fixed point to assert against.
    response = client.get("/reference/seasons/2023/rounds/1/drivers/max_verstappen/form")

    assert response.status_code == 200
    body = response.json()
    assert body["team_id"] == "red_bull"
    assert body["grid_position"] == 1
    assert body["current_driver_wins"] == 0
    assert body["current_driver_podiums"] == 0
    assert body["actual_position"] == 1
    assert body["actual_winner"] is True
    assert isinstance(body["mean_previous_grid"], float)


def test_driver_form_unknown_driver_is_404():
    response = client.get("/reference/seasons/2023/rounds/1/drivers/not_a_real_driver/form")
    assert response.status_code == 404


def test_driver_form_unknown_round_is_404():
    response = client.get("/reference/seasons/2023/rounds/999/drivers/max_verstappen/form")
    assert response.status_code == 404
