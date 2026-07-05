"""Reproducible data/model pipeline for the F1 clustering and prediction models.

Codifies the sequence that's otherwise only implicit in the order of the
A1/A2 (clustering) and B1/B2 (classification) notebooks: raw extraction from
FastF1 -> feature engineering -> preprocessing -> model training. Each stage
is idempotent (skips if its output file already exists) so re-running this
after a docs/notebook change doesn't re-hit the FastF1 API for data that's
already cached on disk.

The notebooks remain the place for EDA and model comparison (trying
different clustering algorithms, comparing classifiers) - this script only
encodes the winning choices already made there: 7 fixed clustering features,
minmax scaling, and a final XGBoost classifier.

Usage
-----
    uv run python scripts/build_pipeline.py                    # run every stage
    uv run python scripts/build_pipeline.py --stage clustering # one group only
    uv run python scripts/build_pipeline.py --force             # recompute
                                                                   features/
                                                                   scaling/
                                                                   training
    uv run python scripts/build_pipeline.py --force-extract      # also
                                                                   re-hit
                                                                   FastF1/
                                                                   Ergast for
                                                                   raw data
    uv run python scripts/build_pipeline.py --quick-train        # tiny param
                                                                   grid, for
                                                                   smoke-testing
                                                                   the script
                                                                   itself, not
                                                                   for a real
                                                                   model

Warning
-------
Raw extraction (races, per-circuit telemetry, per-race results) calls
FastF1/Ergast over the network and can take a long time for the classifier's
results.csv (once per race across every season in range - this is why the
original notebook recommends extracting season by season by hand). These
stages are only re-run when their output file is missing, or when
--force-extract is passed explicitly; a plain --force only recomputes the
cheap, deterministic feature-engineering/preprocessing/training stages.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from src.circuit_clustering_model.extract import extract_races_and_results_dataframes
from src.circuit_clustering_model.feature_engineering import generate_features
from src.preprocess import preprocess
from src.race_prediction_model.classification import ClassificationModels
from src.race_prediction_model.extract import extract_races_dataframe, extract_results_dataframe
from src.race_prediction_model.feature_engineering import add_features_to_results, team_id_replacement

DATA_OUTPUT = "data/output"
DATA_PREPROCESSED = "data/preprocessed"
MODEL_DIR = "model"

RACES_CSV = f"{DATA_OUTPUT}/races.csv"
CIRCUITS_RAW_CSV = f"{DATA_OUTPUT}/circuits.csv"
CIRCUITS_COMPLETE_CSV = f"{DATA_OUTPUT}/circuits_complete.csv"
FEATURED_CIRCUITS_CSV = f"{DATA_OUTPUT}/featured_circuits_complete.csv"
CIRCUITS_SCALED_CSV = f"{DATA_PREPROCESSED}/circuits_scaled.csv"
RESULTS_CSV = f"{DATA_OUTPUT}/results.csv"
FEATURED_RESULTS_CSV = f"{DATA_OUTPUT}/featured_results.csv"

# First season with FastF1 telemetry / Ergast data used across both models.
FIRST_SEASON = 2010
LAST_SEASON = 2024
# Clustering only looks at each circuit's most recent appearance from this
# season onward (2018+ chosen for telemetry-era representativeness, see
# notebook/README.md).
CLUSTERING_FIRST_SEASON = 2018
# Correlation analysis (A1) found these redundant with avg_speed/throttle_perc.
CLUSTERING_DROP_COLUMNS = ["brake_perc", "throttle_perc"]


def _skip_or_run(path, force, build_fn, label):
    if os.path.exists(path) and not force:
        print(f"[skip] {label}: {path} already exists")
        return pd.read_csv(path, index_col=0)
    print(f"[run]  {label} -> {path}")
    return build_fn()


def stage_races(force_extract):
    def build():
        return extract_races_dataframe(FIRST_SEASON, end=LAST_SEASON, save=True)

    if os.path.exists(RACES_CSV) and not force_extract:
        print(f"[skip] races: {RACES_CSV} already exists")
        return pd.read_csv(RACES_CSV)
    print(f"[run]  races -> {RACES_CSV}")
    return build()


def stage_clustering_raw(df_races, force_extract):
    """Pole-lap telemetry per circuit, replacing wet/intermediate sessions
    with the circuit's previous (dry) appearance."""

    def build():
        df_races_recent = df_races.loc[:, ["season", "round", "circuitId"]]
        df_races_recent = df_races_recent[df_races_recent["season"] >= CLUSTERING_FIRST_SEASON]
        df_unique_circuits = df_races_recent.sort_values(by="season").drop_duplicates(
            subset=["circuitId"], keep="last"
        )

        df = extract_races_and_results_dataframes(df_unique_circuits)

        wet_races = df[df["compound"].isin(["WET", "INTERMEDIATE"])].index.to_list()
        df_aux = df_races_recent[df_races_recent["circuitId"].isin(wet_races)]
        penultimate_values = df_aux.groupby("circuitId").nth(-2).reset_index()
        df_ext = extract_races_and_results_dataframes(penultimate_values)

        df_circuits = pd.concat([df[df["compound"].isin(["SOFT", "MEDIUM", "HARD"])], df_ext])
        df_circuits.drop(columns="compound", inplace=True)

        os.makedirs(DATA_OUTPUT, exist_ok=True)
        df.to_csv(CIRCUITS_RAW_CSV)
        return df_circuits

    if os.path.exists(CIRCUITS_COMPLETE_CSV) and not force_extract:
        print(f"[skip] clustering raw: {CIRCUITS_COMPLETE_CSV} already exists")
        return pd.read_csv(CIRCUITS_COMPLETE_CSV, index_col=0)
    print(f"[run]  clustering raw -> {CIRCUITS_RAW_CSV}, {CIRCUITS_COMPLETE_CSV}")
    df_circuits = build()
    df_circuits.drop(columns=CLUSTERING_DROP_COLUMNS, inplace=True)
    df_circuits.to_csv(CIRCUITS_COMPLETE_CSV)
    return df_circuits


def stage_clustering_features(df_circuits, force):
    def build():
        return generate_features(df_circuits)

    return _skip_or_run(FEATURED_CIRCUITS_CSV, force, lambda: _save(build(), FEATURED_CIRCUITS_CSV), "clustering features")


def stage_clustering_scale(df_featured, force):
    def build():
        _, df_scaled = preprocess(df_featured, encoding_methods={}, scaling_method="minmax")
        return df_scaled

    os.makedirs(DATA_PREPROCESSED, exist_ok=True)
    return _skip_or_run(CIRCUITS_SCALED_CSV, force, lambda: _save(build(), CIRCUITS_SCALED_CSV), "clustering scaling")


def stage_classification_raw(df_races, force_extract):
    def build():
        results, _sessions = extract_results_dataframe(df_races, save=True)
        return results

    if os.path.exists(RESULTS_CSV) and not force_extract:
        print(f"[skip] classification raw: {RESULTS_CSV} already exists")
        return pd.read_csv(RESULTS_CSV, index_col=0)
    print(f"[run]  classification raw -> {RESULTS_CSV} (this calls FastF1 per race, can be slow)")
    return build()


def stage_classification_features(df_results, force):
    def build():
        df = df_results.copy()
        df.dropna(subset=["GridPosition"], inplace=True)
        team_id_replacement(df)
        add_features_to_results(df)
        df.drop(columns=["season", "CurrentDriverPoints", "CurrentTeamPoints"], inplace=True)
        return df

    return _skip_or_run(
        FEATURED_RESULTS_CSV, force, lambda: _save(build(), FEATURED_RESULTS_CSV), "classification features"
    )


def stage_classification_train(df_results, force, quick_train=False):
    """Encodes/scales the featured results and trains the production XGBoost
    model. Mirrors notebook B2's final cell exactly (single fixed target,
    no balancing) - the model comparison and class-balancing experiments
    stay in the notebook since they're exploratory, not part of the
    reproducible pipeline."""

    best_model_path = f"{MODEL_DIR}/best_model.pkl"
    if os.path.exists(best_model_path) and not force:
        print(f"[skip] classification train: {best_model_path} already exists")
        return

    print(f"[run]  classification train -> {MODEL_DIR}/best_model.pkl, encoder.pkl, scaler.pkl")

    target = "Winner"
    df = df_results.drop(columns=["Position", "Time", "Status", "Points", "Podium"])

    encoding_methods = {
        "onehot": [],
        "target": ["DriverId", "TeamId"],
        "ordinal": {"circuitId": df["circuitId"].unique().tolist()},
        "frequency": [],
    }
    _df_encoded, df_scaled = preprocess(
        df, encoding_methods, "minmax", target_variable=target, save_objects=True
    )

    models = ClassificationModels(df_scaled, target)
    param_grid = {"n_estimators": [100]} if quick_train else None
    models.fit_model("xgboost", param_grid=param_grid, file_name="best_model", cross_validation=10)


def _save(df, path):
    df.to_csv(path)
    return df


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--stage",
        choices=["all", "clustering", "classification"],
        default="all",
        help="Which group of stages to run (default: all).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recompute feature-engineering/preprocessing/training stages even if their output already exists.",
    )
    parser.add_argument(
        "--force-extract",
        action="store_true",
        help="Also re-run raw extraction (races/telemetry/results) against FastF1/Ergast. Implies --force.",
    )
    parser.add_argument(
        "--quick-train",
        action="store_true",
        help="Use a 1-combination param grid for the classifier - for smoke-testing the script, not a real model.",
    )
    args = parser.parse_args()
    force = args.force or args.force_extract

    df_races = stage_races(args.force_extract)

    if args.stage in ("all", "clustering"):
        df_circuits = stage_clustering_raw(df_races, args.force_extract)
        df_featured = stage_clustering_features(df_circuits, force)
        stage_clustering_scale(df_featured, force)

    if args.stage in ("all", "classification"):
        df_results = stage_classification_raw(df_races, args.force_extract)
        df_featured_results = stage_classification_features(df_results, force)
        stage_classification_train(df_featured_results, force, quick_train=args.quick_train)

    print("Pipeline done.")


if __name__ == "__main__":
    main()
