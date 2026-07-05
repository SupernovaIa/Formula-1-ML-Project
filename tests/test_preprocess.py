import pandas as pd
import pytest

from src.preprocess import Encoding, preprocess, scale_df


def _sample_df():
    return pd.DataFrame({
        "circuit": ["monza", "spa", "monza", "monaco", "spa"],
        "team": ["ferrari", "mercedes", "ferrari", "red_bull", "mercedes"],
        "grid": [1, 2, 3, 4, 5],
        "winner": [1, 0, 0, 1, 0],
    })


class TestEncoding:
    def test_one_hot_encoding_creates_dummy_columns_and_drops_original(self):
        df = _sample_df()
        encoder = Encoding(df, {"onehot": ["team"]}, target_variable="winner")
        encoder.execute_all_encodings()

        assert "team" not in encoder.df.columns
        assert {"team_ferrari", "team_mercedes", "team_red_bull"} <= set(encoder.df.columns)
        assert encoder.df["team_ferrari"].sum() == 2

    def test_ordinal_encoding_respects_given_category_order(self):
        df = _sample_df()
        order = ["monaco", "spa", "monza"]
        encoder = Encoding(df, {"ordinal": {"circuit": order}}, target_variable="winner")
        encoder.execute_all_encodings()

        assert encoder.df.loc[0, "circuit"] == order.index("monza")
        assert encoder.df.loc[1, "circuit"] == order.index("spa")

    def test_ordinal_encoding_missing_column_raises(self):
        df = _sample_df()
        encoder = Encoding(df, {"ordinal": {"nonexistent": ["a", "b"]}}, target_variable="winner")
        with pytest.raises(ValueError):
            encoder._ordinal_encoding()

    def test_frequency_encoding_maps_to_relative_frequency(self):
        df = _sample_df()
        encoder = Encoding(df, {"frequency": ["team"]}, target_variable="winner")
        encoder.execute_all_encodings()

        assert encoder.df["team"].iloc[0] == pytest.approx(2 / 5)  # ferrari: 2 of 5
        assert encoder.df["team"].iloc[1] == pytest.approx(2 / 5)  # mercedes: 2 of 5

    def test_target_encoding_without_target_variable_is_caught_not_raised(self):
        df = _sample_df()
        encoder = Encoding(df, {"target": ["team"]}, target_variable="missing_column")
        # execute_all_encodings wraps each step in try/except, so a bad target
        # variable shouldn't blow up the whole pipeline.
        encoder.execute_all_encodings()
        assert "team" in encoder.df.columns

    def test_transform_applies_fitted_ordinal_encoder_to_new_data(self):
        df = _sample_df()
        order = ["monaco", "spa", "monza"]
        encoder = Encoding(df, {"ordinal": {"circuit": order}}, target_variable="winner")
        encoder.execute_all_encodings()

        new_data = pd.DataFrame({"circuit": ["spa"], "team": ["mercedes"], "grid": [1], "winner": [0]})
        transformed = encoder.transform(new_data)
        assert transformed.loc[0, "circuit"] == order.index("spa")


class TestScaleDf:
    def test_minmax_scales_into_zero_one_range(self):
        df = _sample_df()
        scaled, scaler = scale_df(df, ["grid"], method="minmax")
        assert scaled["grid"].min() == 0.0
        assert scaled["grid"].max() == 1.0

    def test_include_others_keeps_unscaled_columns(self):
        df = _sample_df()
        scaled, _ = scale_df(df, ["grid"], method="minmax", include_others=True)
        assert "team" in scaled.columns
        assert "circuit" in scaled.columns

    def test_invalid_method_raises(self):
        df = _sample_df()
        with pytest.raises(ValueError):
            scale_df(df, ["grid"], method="not_a_method")

    def test_missing_column_raises(self):
        df = _sample_df()
        with pytest.raises(ValueError):
            scale_df(df, ["nonexistent"], method="minmax")


class TestPreprocess:
    def test_with_target_variable_scales_every_column_except_target(self):
        df = _sample_df()
        encoding_methods = {"ordinal": {"circuit": ["monaco", "spa", "monza"]}, "frequency": ["team"]}
        _df_encoded, df_scaled = preprocess(df, encoding_methods, "minmax", target_variable="winner")

        assert df_scaled["winner"].tolist() == df["winner"].tolist()  # target left untouched
        assert df_scaled["grid"].min() == 0.0 and df_scaled["grid"].max() == 1.0

    def test_without_target_variable_does_not_raise(self):
        # Regression test: preprocess() used to crash on df.drop(columns=None)
        # when no target_variable was passed - the clustering pipeline's exact
        # call pattern (see scripts/build_pipeline.py's clustering-scale stage).
        df = _sample_df().drop(columns=["winner", "circuit", "team"])
        _df_encoded, df_scaled = preprocess(df, {}, "minmax")
        assert df_scaled["grid"].min() == 0.0 and df_scaled["grid"].max() == 1.0

    def test_columns_drop_removes_columns_before_encoding(self):
        df = _sample_df()
        df_encoded, _ = preprocess(
            df, {}, "minmax", columns_drop=["team", "circuit"], target_variable="winner"
        )
        assert "team" not in df_encoded.columns and "circuit" not in df_encoded.columns
