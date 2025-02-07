import streamlit as st
import fastf1
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.metrics import silhouette_score

from src.dashboard.season import *

# Data processing  
# -----------------------------------------------------------------------
import pandas as pd
import numpy as np

# Clusters and metrics
# -----------------------------------------------------------------------
from yellowbrick.cluster import KElbowVisualizer

# Clustering models
# -----------------------------------------------------------------------
from sklearn.cluster import KMeans

# Custom functions and classes
# -----------------------------------------------------------------------
from src.preprocess import preprocess # Remove this if not needed
from src.circuit_clustering_model.clustering import *




st.set_page_config(page_title="F1 Season report", page_icon="🏎️", layout="wide")
st.title("🏎️ Clustering de circuitos WEEEE")


with st.sidebar:

    if 'df' not in st.session_state:
        st.session_state.df = None

    if 'trained' not in st.session_state:
        st.session_state.trained = False

    if st.button("Load Data"):
        with st.spinner("Loading race data..."):

            st.session_state.df = pd.read_csv('data/output/featured_circuits_complete.csv', index_col=0)
            st.session_state.df_scaled = pd.read_csv('data/preprocessed/circuits_scaled.csv', index_col=0)
            st.write('Data loaded')


if st.session_state.df is not None:

    st.dataframe(st.session_state.df.head())

    st.write("Welcome to F1 fantasy cluster")
    # Aquí hay que explicar las columnas que podemos seleccionar
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

    # Crear DataFrame con resultados
    df_results = pd.DataFrame({"Número de Clusters (k)": k_values, "Puntuación Silhouette": silhouette_scores})

    # Ordenar por los mejores 3 valores de Silhouette Score
    top_3 = df_results.nlargest(3, "Puntuación Silhouette")  # Selecciona los 3 mejores

    # Graficar con Plotly
    fig = px.line(df_results, x="Número de Clusters (k)", y="Puntuación Silhouette", markers=True,
                title="Selección de los 3 Mejores k por Silhouette Score",
                labels={"Número de Clusters (k)": "Número de Clusters (k)", "Puntuación Silhouette": "Silhouette Score"})

    # Agregar los 3 mejores puntos en el gráfico
    for _, row in top_3.iterrows():
        fig.add_scatter(x=[row["Número de Clusters (k)"]], y=[row["Puntuación Silhouette"]], mode="markers",
                        marker=dict(size=10, color="red"), name=f"k={int(row['Número de Clusters (k)'])}")

    # Mostrar en Streamlit
    st.plotly_chart(fig)

    # Mostrar los 3 mejores valores en Streamlit
    st.write("### 📌 Los 3 mejores valores de `k` según Silhouette Score:")
    for i, row in top_3.iterrows():
        st.write(f"🔹 k={int(row['Número de Clusters (k)'])} con Silhouette Score: {row['Puntuación Silhouette']:.4f}")

    with st.sidebar:
        n_clusters = st.slider("Select number of clusters:", 2, 12, 7, 1)
        
    model_kmeans = KMeans(n_clusters=n_clusters, random_state=42)

    km_fit = model_kmeans.fit(st.session_state.df_kmeans)
    labels = km_fit.labels_

    st.session_state.df_kmeans['cluster'] = km_fit.labels_
    st.session_state.df['cluster'] = km_fit.labels_

    st.session_state.trained = True

    # Definir colores para los clusters
    import plotly.colors as pc

    with st.expander('Show clustering results'):
        # Obtener una paleta de colores de Plotly
        cmap = pc.qualitative.Prism  # Puedes cambiarlo por otra como "Dark24", "Set3", etc.

        # Mapear colores únicos para cada cluster
        unique_clusters = sorted(st.session_state.df_kmeans["cluster"].unique())
        cluster_colors = {cluster: cmap[i % len(cmap)] for i, cluster in enumerate(unique_clusters)}


        # Mostrar los circuitos por cluster con etiquetas de colores
        st.write("### 🏎️ Clasificación de circuitos por cluster")
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
        # Poner un selector de variables
        fig = plot_clusters(st.session_state.df, 'straight_prop')
        st.plotly_chart(fig)


    elif viz_type == "Scatter":
        # Poner un selector de columnas
        col1 = 'slow_corners_prop'
        col2 = 'straight_prop'
        fig = plot_cluster_scatter(st.session_state.df_kmeans, col1, col2)
        st.plotly_chart(fig)


    elif viz_type == "Radar":
        fig = plot_radar(st.session_state.df_kmeans, columns=st.session_state.df_kmeans.drop(columns='cluster').columns, opacity=0.3)
        st.plotly_chart(fig)


    elif viz_type == "PCA":
        pca_df, pca_evr = get_pca(st.session_state.df_kmeans.drop(columns='cluster'))
        pca_df = pd.concat([pca_df, st.session_state.df_kmeans['cluster'].reset_index()], axis=1).set_index('index')

        # Show total explained variance ratio
        # print("Explained variance for each component:", pca_evr)
        # print("Total explained variance:", round(sum(pca_evr), 4))

        fig = plot_cluster_scatter(pca_df, marker_size=15, *pca_df.columns.drop('cluster').to_list())

        st.plotly_chart(fig)

