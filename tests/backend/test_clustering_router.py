from fastapi.testclient import TestClient

from backend.main import app

client = TestClient(app)


def test_columns_lists_the_fixed_clustering_features():
    response = client.get("/clustering/columns")

    assert response.status_code == 200
    body = response.json()
    assert set(body["clustering_features"]) <= set(body["all"])


def test_silhouette_scores_returns_one_entry_per_k():
    response = client.get("/clustering/silhouette-scores", params={"k_min": 2, "k_max": 5})

    assert response.status_code == 200
    scores = response.json()
    assert [s["k"] for s in scores] == [2, 3, 4]
    assert all(-1.0 <= s["silhouette_score"] <= 1.0 for s in scores)


def test_assignments_returns_a_cluster_label_per_circuit():
    response = client.get("/clustering/assignments", params={"n_clusters": 7})

    assert response.status_code == 200
    rows = response.json()
    assert len(rows) == 30  # circuits in data/preprocessed/circuits_scaled.csv
    assert {row["cluster"] for row in rows} <= set(range(7))


def test_assignments_is_deterministic_for_a_given_n_clusters():
    first = client.get("/clustering/assignments", params={"n_clusters": 7}).json()
    second = client.get("/clustering/assignments", params={"n_clusters": 7}).json()
    assert first == second


def test_plot_scatter_returns_a_plotly_figure():
    response = client.get(
        "/clustering/plot/scatter", params={"col1": "avg_speed", "col2": "straight_prop"}
    )

    assert response.status_code == 200
    figure = response.json()
    assert "data" in figure and "layout" in figure
