import pandas as pd

from src.race_prediction_model.feature_engineering import add_features_to_results, team_id_replacement


def test_team_id_replacement_maps_historical_teams_to_current_entity():
    df = pd.DataFrame({"TeamId": ["renault", "lotus_f1", "toro_rosso", "ferrari"]})
    team_id_replacement(df)

    assert df["TeamId"].tolist() == ["alpine", "alpine", "rb", "ferrari"]


def _sample_results():
    # Two drivers, two seasons, three rounds each, driver "alonso" wins round 3
    # of season 2020 - values chosen so the rolling/cumulative math is easy to
    # check by hand.
    return pd.DataFrame({
        "season": [2020, 2020, 2020, 2020, 2020, 2020],
        "round": [1, 1, 2, 2, 3, 3],
        "DriverId": ["alonso", "hamilton", "alonso", "hamilton", "alonso", "hamilton"],
        "TeamId": ["alpine", "mercedes", "alpine", "mercedes", "alpine", "mercedes"],
        "Position": [3, 1, 2, 1, 1, 2],
        "GridPosition": [4, 1, 3, 1, 2, 1],
        "Points": [15, 25, 18, 25, 25, 18],
    })


def test_add_features_to_results_flags_winner_and_podium():
    df = _sample_results()
    add_features_to_results(df)

    winners = df[df["Winner"] == 1][["DriverId", "round"]].values.tolist()
    assert ["hamilton", 1] in winners
    assert ["alonso", 3] in winners
    assert df["Podium"].sum() == len(df)  # every finisher here is top 3


def test_add_features_to_results_rolling_mean_is_trailing_and_includes_current_race():
    # Despite the "Previous" in the name, MeanPreviousGrid is a plain trailing
    # rolling mean that includes the current row - not shifted by one race.
    df = _sample_results()
    add_features_to_results(df, window=3)

    alonso = df[df["DriverId"] == "alonso"].sort_values("round")
    assert alonso.iloc[0]["MeanPreviousGrid"] == 4  # round 1: mean([4])
    assert alonso.iloc[1]["MeanPreviousGrid"] == (4 + 3) / 2  # round 2: mean([4, 3])
    assert alonso.iloc[2]["MeanPreviousGrid"] == (4 + 3 + 2) / 3  # round 3: mean([4, 3, 2])


def test_add_features_to_results_cumulative_points_exclude_current_race():
    df = _sample_results()
    add_features_to_results(df)

    alonso = df[df["DriverId"] == "alonso"].sort_values("round")
    assert alonso.iloc[0]["CurrentDriverPoints"] == 0
    assert alonso.iloc[1]["CurrentDriverPoints"] == 15
    assert alonso.iloc[2]["CurrentDriverPoints"] == 15 + 18
