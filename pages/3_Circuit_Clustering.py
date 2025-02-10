# Web application framework and plotting utilities
# -----------------------------------------------------------------------
import streamlit as st
import plotly.express as px
import plotly.colors as pc

# Metrics for clustering evaluation
# -----------------------------------------------------------------------
from sklearn.metrics import silhouette_score

# Dashboard modules
# -----------------------------------------------------------------------
from src.dashboard.season import *

# Data processing
# -----------------------------------------------------------------------
import pandas as pd
import numpy as np

# Clustering models
# -----------------------------------------------------------------------
from sklearn.cluster import KMeans

# Custom functions and classes
# -----------------------------------------------------------------------
from src.preprocess import preprocess  # Remove this if not needed
from src.circuit_clustering_model.clustering import *


st.set_page_config(page_title="F1 Circuit clustering", page_icon="🏎️", layout="wide")
st.title("Magic circuit cluster 🪄")

with st.sidebar:

    if 'df' not in st.session_state:
        st.session_state.clear()
        st.session_state.df = None

    if 'trained' not in st.session_state:
        st.session_state.trained = False

    if st.button("Load Data"):
        with st.spinner("Loading race data..."):

            st.session_state.df = pd.read_csv('data/output/featured_circuits_complete.csv', index_col=0)
            st.session_state.df_scaled = pd.read_csv('data/preprocessed/circuits_scaled.csv', index_col=0)
            st.write('Data loaded')

if st.session_state.df is not None:

    st.header("Welcome to F1 fantasy cluster ✨")
    st.dataframe(st.session_state.df.head())
    # Select features
    # columns = st.multiselect('Select the features you want to use for clustering', st.session_state.df_scaled.columns.to_list())
    columns = ['avg_speed', 'straight_prop', 'slow_corners_prop']

    st.session_state.df_kmeans = st.session_state.df_scaled[columns]

    k_values = list(range(2, 12))
    silhouette_scores = []

    for k in k_values:
        model = KMeans(n_clusters=k, random_state=42)
        labels = model.fit_predict(st.session_state.df_kmeans)
        score = silhouette_score(st.session_state.df_kmeans, labels)
        silhouette_scores.append(score)

    df_results = pd.DataFrame({"Number of clusters (k)": k_values, "Silhouette Score": silhouette_scores})

    top_3 = df_results.nlargest(3, "Silhouette Score")

    fig = px.line(df_results, x="Number of clusters (k)", y="Silhouette Score", markers=True,
                title="Selection of the Top 3 k by Silhouette Score.",
                labels={"Number of clusters (k)": "Number of clusters (k)", "Silhouette Score": "Silhouette Score"})

    for _, row in top_3.iterrows():
        fig.add_scatter(x=[row["Number of clusters (k)"]], y=[row["Silhouette Score"]], mode="markers",
                        marker=dict(size=10, color="red"), name=f"k={int(row['Number of clusters (k)'])}")

    st.plotly_chart(fig)

    st.write("### 📌 The Top 3 `k` Values Based on Silhouette Score:")
    for i, row in top_3.iterrows():
        st.write(f"🔹 k={int(row['Number of clusters (k)'])} with Silhouette Score: {row['Silhouette Score']:.3f}")

    with st.sidebar:
        n_clusters = st.slider("Select number of clusters:", 2, 12, 7, 1)
        
    model_kmeans = KMeans(n_clusters=n_clusters, random_state=42)

    km_fit = model_kmeans.fit(st.session_state.df_kmeans)
    labels = km_fit.labels_

    st.session_state.df_kmeans['cluster'] = km_fit.labels_
    st.session_state.df['cluster'] = km_fit.labels_

    st.session_state.trained = True

    with st.expander('Show clustering results'):

        # Map unique colors for each cluster.
        cmap = pc.qualitative.Prism
        unique_clusters = sorted(st.session_state.df_kmeans["cluster"].unique())
        cluster_colors = {cluster: cmap[i % len(cmap)] for i, cluster in enumerate(unique_clusters)}

        # Display the circuits by cluster with color-coded labels.
        st.write("### 🏎️ Classification of circuits by cluster.")
        for circuito, row in st.session_state.df_kmeans.sort_values(by='cluster').iterrows():
            color = cluster_colors[int(row["cluster"])]
            st.markdown(f'<div style="display: inline-block; background-color: {color}; padding: 5px 10px; border-radius: 10px; margin: 5px; color: white;">'
                        f'<b>{circuito.replace("_", " ").title()} (Cluster {int(row["cluster"])})</b></div>', 
                        unsafe_allow_html=True)
            
if st.session_state.trained:

    with st.sidebar:
        options = ["Clusters", "Scatter", "Radar", "PCA"]
        viz_type = st.selectbox("Select viz type", options)

    if viz_type == "Clusters":
        col = st.selectbox("Select variable", st.session_state.df.columns.to_list())
        fig = plot_clusters(st.session_state.df, col)
        st.plotly_chart(fig)

    elif viz_type == "Scatter":
        col1 = st.selectbox("Select variable", st.session_state.df_kmeans.drop(columns=['cluster']).columns.to_list())
        col2 = st.selectbox("Select variable", st.session_state.df_kmeans.drop(columns=[col1, 'cluster']).columns.to_list())
        size = st.slider("Marker size", min_value=1, max_value=30, value=15, step=1)
        fig = plot_cluster_scatter(st.session_state.df_kmeans, col1, col2, marker_size=size)
        st.plotly_chart(fig)

    elif viz_type == "Radar":
        fig = plot_radar(st.session_state.df_kmeans, columns=st.session_state.df_kmeans.drop(columns='cluster').columns, opacity=0.3)
        st.plotly_chart(fig)

    elif viz_type == "PCA":
        pca_df, pca_evr = get_pca(st.session_state.df_kmeans.drop(columns='cluster'))
        pca_df = pd.concat([pca_df, st.session_state.df_kmeans['cluster'].reset_index()], axis=1).set_index('index')
        size = st.slider("Marker size", min_value=1, max_value=30, value=15, step=1)
        fig = plot_cluster_scatter(pca_df, marker_size=size, *pca_df.columns.drop('cluster').to_list())
        st.plotly_chart(fig)

