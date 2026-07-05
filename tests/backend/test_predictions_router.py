import pytest
from fastapi.testclient import TestClient

from backend.main import app

client = TestClient(app)


def _payload(**overrides):
    payload = {
        "driver_id": "verstappen",
        "team_id": "red_bull",
        "circuit_id": "monza",
        "grid_position": 1,
        "round_number": 15,
        "mean_previous_grid": 1.5,
        "mean_previous_position": 1.2,
        "current_driver_wins": 8,
        "current_driver_podiums": 12,
    }
    payload.update(overrides)
    return payload


def test_predict_winner_returns_a_probability_between_zero_and_one():
    response = client.post("/predictions/winner", json=_payload())

    assert response.status_code == 200
    body = response.json()
    assert set(body.keys()) == {"predicted_winner", "win_probability"}
    assert 0.0 <= body["win_probability"] <= 1.0
    assert isinstance(body["predicted_winner"], bool)


def test_predict_winner_favors_pole_position_over_the_back_of_the_grid():
    front = client.post("/predictions/winner", json=_payload(grid_position=1)).json()
    back = client.post("/predictions/winner", json=_payload(grid_position=20)).json()

    assert front["win_probability"] > back["win_probability"]


@pytest.mark.parametrize("missing_field", ["driver_id", "grid_position", "circuit_id"])
def test_predict_winner_missing_required_field_is_422(missing_field):
    payload = _payload()
    del payload[missing_field]

    response = client.post("/predictions/winner", json=payload)
    assert response.status_code == 422


def test_predict_winner_unknown_circuit_does_not_crash():
    # circuitId goes through an OrdinalEncoder with handle_unknown="use_encoded_value",
    # so an unseen circuit should degrade gracefully (NaN feature), not 500.
    response = client.post("/predictions/winner", json=_payload(circuit_id="not_a_real_circuit"))
    assert response.status_code == 200
