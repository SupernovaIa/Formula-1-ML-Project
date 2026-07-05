import pandas as pd

from src.circuit_clustering_model.feature_engineering import generate_features


def _sample_circuits():
    return pd.DataFrame(
        {
            "n_gear1_corners": [2, 0],
            "n_gear2_corners": [1, 0],
            "n_gear3_corners": [1, 1],
            "n_gear4_corners": [0, 1],
            "n_gear5_corners": [1, 1],
            "n_gear6_corners": [1, 0],
            "n_gear7_corners": [0, 1],
            "n_gear8_corners": [0, 0],
            "n_slow_corners": [3, 1],
            "n_medium_corners": [2, 2],
            "n_fast_corners": [1, 1],
            "n_corners": [6, 4],
            "straight_length": [1000, 500],
            "distance": [5000, 4000],
            "gear_changes": [40, 30],
            "laptime": [80.0, 70.0],
        },
        index=["monza", "monaco"],
    )


def test_generate_features_computes_expected_proportions():
    df = _sample_circuits()
    out = generate_features(df)

    assert out.loc["monza", "short_gear_corners_prop"] == 4 / 6
    assert out.loc["monza", "long_gear_corners_prop"] == 2 / 6
    assert out.loc["monza", "slow_corners_prop"] == 3 / 6
    assert out.loc["monza", "straight_prop"] == 1000 / 5000
    assert out.loc["monza", "gear_changes_per_km"] == 40 / 5000 * 1000
    assert out.loc["monza", "n_corners_per_km"] == 6 / 5000 * 1000


def test_generate_features_drops_source_columns():
    df = _sample_circuits()
    out = generate_features(df)

    for col in ["n_gear1_corners", "gear_changes", "straight_length", "n_corners", "laptime"]:
        assert col not in out.columns


def test_generate_features_handles_zero_corners_without_dividing_by_zero():
    df = _sample_circuits()
    df.loc["monza", ["n_gear1_corners", "n_gear2_corners", "n_gear3_corners", "n_gear4_corners",
                      "n_gear5_corners", "n_gear6_corners", "n_gear7_corners", "n_gear8_corners",
                      "n_slow_corners", "n_medium_corners", "n_fast_corners", "n_corners"]] = 0

    out = generate_features(df)

    assert out.loc["monza", "short_gear_corners_prop"] == 0
    assert out.loc["monza", "slow_corners_prop"] == 0
