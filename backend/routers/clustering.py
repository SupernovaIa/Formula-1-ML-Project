import pandas as pd
from fastapi import APIRouter
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from backend.core.serialization import df_to_records, fig_to_json
from src.circuit_clustering_model.clustering import get_pca, plot_cluster_scatter, plot_clusters, plot_radar

router = APIRouter(prefix="/clustering", tags=["clustering"])

FEATURES = ["avg_speed", "straight_prop", "slow_corners_prop"]


def _load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv("data/output/featured_circuits_complete.csv", index_col=0)
    df_scaled = pd.read_csv("data/preprocessed/circuits_scaled.csv", index_col=0)
    return df, df_scaled[FEATURES]


def _fit(n_clusters: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    df, df_kmeans = _load_data()
    df_kmeans = df_kmeans.copy()

    model = KMeans(n_clusters=n_clusters, random_state=42)
    labels = model.fit_predict(df_kmeans)

    df_kmeans["cluster"] = labels
    df = df.copy()
    df["cluster"] = labels

    return df, df_kmeans


@router.get("/columns")
def columns():
    df, _ = _load_data()
    return {"all": df.columns.tolist(), "clustering_features": FEATURES}


@router.get("/silhouette-scores")
def silhouette_scores(k_min: int = 2, k_max: int = 12):
    _, df_kmeans = _load_data()
    scores = []
    for k in range(k_min, k_max):
        labels = KMeans(n_clusters=k, random_state=42).fit_predict(df_kmeans)
        scores.append({"k": k, "silhouette_score": silhouette_score(df_kmeans, labels)})
    return scores


@router.get("/assignments")
def assignments(n_clusters: int = 7):
    # Raw (unscaled) feature values - the model itself still fits on the
    # scaled ones (see _fit), but callers displaying these want real
    # km/h and proportions, not MinMax-scaled 0-1 ranks.
    df, _ = _fit(n_clusters)
    return df_to_records(df[FEATURES + ["cluster"]].reset_index())


@router.get("/plot/mean-by-cluster")
def plot_mean_by_cluster(column: str, n_clusters: int = 7):
    df, _ = _fit(n_clusters)
    return fig_to_json(plot_clusters(df, column))


@router.get("/plot/scatter")
def plot_scatter(col1: str, col2: str, n_clusters: int = 7, marker_size: int = 15):
    _, df_kmeans = _fit(n_clusters)
    return fig_to_json(plot_cluster_scatter(df_kmeans, col1, col2, marker_size=marker_size))


@router.get("/plot/radar")
def plot_radar_chart(n_clusters: int = 7, opacity: float = 0.3):
    _, df_kmeans = _fit(n_clusters)
    return fig_to_json(plot_radar(df_kmeans, columns=FEATURES, opacity=opacity))


@router.get("/plot/pca")
def plot_pca(n_clusters: int = 7, marker_size: int = 15):
    _, df_kmeans = _fit(n_clusters)
    pca_df, _explained_variance = get_pca(df_kmeans.drop(columns="cluster"))
    pca_df["cluster"] = df_kmeans["cluster"].to_numpy()
    return fig_to_json(plot_cluster_scatter(pca_df, "PC1", "PC2", marker_size=marker_size))
