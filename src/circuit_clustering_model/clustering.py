# Data processing
# -----------------------------------------------------------------------
import numpy as np
import pandas as pd
import math

# For visualizations
# -----------------------------------------------------------------------
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px

# Extracting the number of clusters and metrics
# -----------------------------------------------------------------------
from yellowbrick.cluster import KElbowVisualizer
from sklearn.metrics import silhouette_score, davies_bouldin_score

# Clustering models
# -----------------------------------------------------------------------
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN

# For PCA analysis
# -----------------------------------------------------------------------
from sklearn.decomposition import PCA

# For visualizing dendrograms
# -----------------------------------------------------------------------
import scipy.cluster.hierarchy as sch


def clustering_metrics(df, labels):
    """
    Calculate clustering metrics and return them in a DataFrame.

    Parameters
    ----------
        - df (pd.DataFrame): The data used for clustering.
        - labels (array-like): Cluster labels assigned to each data point.

    Returns
    -------
        - (pd.DataFrame): A DataFrame containing the silhouette score, Davies-Bouldin index, and cluster cardinality for each cluster.
    """

    silhouette = silhouette_score(df, labels)
    davies_bouldin = davies_bouldin_score(df, labels)

    cardinality = {cluster: sum(labels == cluster) for cluster in np.unique(labels)}

    df_metrics = pd.DataFrame({
                        "silhouette_score": silhouette,
                        "davies_bouldin_index": davies_bouldin,
                        "cardinality": cardinality
                        })
    
    return df_metrics


def plot_cluster_scatter(df, col1, col2, marker_size=10):
    """
    Plots a scatter plot for a clustering model using Plotly.
    
    Parameters
    ----------
        - df (pd.DataFrame): DataFrame containing clustering results with a 'cluster' column.
        - col1 (str): Column name for the x-axis.
        - col2 (str): Column name for the y-axis.
        - marker_size (int, optional): Size of markers. Default is 10.
        
    Returns
    -------
        - fig (plotly.graph_objects.Figure): The generated scatter plot.
    """
    
    if 'cluster' not in df.columns or col1 not in df.columns or col2 not in df.columns:
        raise ValueError("Ensure that 'cluster', '{}' and '{}' exist in the dataframe.".format(col1, col2))
    
    cluster_labels = sorted(df['cluster'].unique())  # Get unique cluster labels
    
    # Get Plotly color palette
    cmap = px.colors.qualitative.Prism
    n_colors = len(cmap)

    fig = go.Figure()

    # Iterate over unique clusters
    for i, cluster_label in enumerate(cluster_labels):
        cluster_data = df[df['cluster'] == cluster_label]
        
        fig.add_trace(go.Scatter(
            x=cluster_data[col1], 
            y=cluster_data[col2], 
            mode='markers',
            marker=dict(size=marker_size, color=cmap[i % n_colors], line=dict(width=1, color='black')),
            name=f'Cluster {cluster_label}',
            text=[f"Circuit: {idx}<br>Cluster: {cluster_label}<br>{col1}: {cluster_data[col1].iloc[j].round(4)}<br>{col2}: {cluster_data[col2].iloc[j].round(4)}"
                  for j, idx in enumerate(cluster_data.index)],
            hoverinfo="text"
        ))

    # Configure layout
    fig.update_layout(
        title="Clusters of Circuits (K-Means Clustering Model)",
        xaxis_title=col1,
        yaxis_title=col2,
        template="plotly_dark",
        legend_title="Clusters"
    )

    return fig


def plot_radar(df, columns, opacity=0.5):
    """
    Generates a radar plot to visualize cluster profiles based on specified columns.

    Parameters
    -----------
    - df (pd.DataFrame): The input DataFrame containing the data and cluster labels.
    - columns (list of str): The list of column names to include in the radar plot.
    - opacity (float, optional): The transparency level of the radar plot areas. Defaults to 0.5.

    Returns
    --------
    - (plotly.graph_objects.Figure): A Plotly radar chart visualizing the cluster profiles.
    """

    # Get mean values for every column by cluster
    cluster_means = df.groupby('cluster')[columns].mean()

    # Close the radar chart by repeating the first column's values
    cluster_means = cluster_means.T  # Transpose for easier manipulation
    cluster_means.loc["first_column"] = cluster_means.iloc[0]  # Repeat first row
    cluster_means = cluster_means.T  # Transpose back

    # Define angles for the radar chart
    num_vars = len(columns)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]  # Close the loop

    # Create figure
    fig = go.Figure()

    # Add each cluster to the plot
    for i, row in cluster_means.iterrows():
        fig.add_trace(go.Scatterpolar(
            r=row.values,
            theta=columns + [columns[0]],  # Close loop
            fill='toself',
            name=f'Cluster {i}',
            opacity=opacity
        ))

    # Configure layout
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True)
        ),
        title="Cluster radar chart",
        template="plotly_dark",
        height=800
    )

    return fig


def plot_clusters(df, col):
    """
    Plots the mean value of a specified column for each cluster.

    Parameters
    ----------
        - df (DataFrame): DataFrame containing the clustering results.
        - col (str): Column name to compute mean values per cluster.

    Returns
    -------
        - fig (plotly.graph_objects.Figure): Bar chart of mean values per cluster.
    """
    if 'cluster' not in df.columns or col not in df.columns:
        raise ValueError(f"Ensure 'cluster' and '{col}' exist in the DataFrame.")

    # Ensure column is numeric
    if not pd.api.types.is_numeric_dtype(df[col]):
        raise ValueError(f"Column '{col}' must be numeric.")

    # Compute mean values per cluster
    df_group = df.groupby('cluster', as_index=False)[col].mean().sort_values('cluster')

    # Define color palette
    colors = px.colors.qualitative.Prism
    num_colors = len(colors)

    # Create figure
    fig = go.Figure()

    # Add a bar for each cluster
    for i, row in df_group.iterrows():
        fig.add_trace(go.Bar(
            x=[f'Cluster {int(row["cluster"])}'],
            y=[row[col]],  
            name=f'Cluster {int(row["cluster"])}',  
            marker_color=colors[i % num_colors]
        ))

    # Configure layout
    fig.update_layout(
        title=f"Cluster Comparison - {col}",
        xaxis_title="Clusters",
        yaxis_title="Mean Value",
        template="plotly_dark",
        showlegend=True,
        legend_title_text="Clusters",
        bargap=0.2  # Adjust bar spacing
    )

    return fig


def get_pca(df, n=2, prefix="PC"):
    """
    Performs PCA transformation on a dataframe (assumes it's already scaled).
    
    Parameters:
    - df (pd.DataFrame): Scaled numerical dataframe.
    - n (int): Number of principal components to keep.
    - prefix (str): Prefix for component column names (default: "PC").
    
    Returns:
    - pca_df (pd.DataFrame): Dataframe with transformed principal components.
    - explained_variance (list): Variance explained by each component.
    """
    
    if n > df.shape[1]:
        raise ValueError(f"n_components ({n}) cannot exceed the number of features ({df.shape[1]}).")
    
    # Apply PCA
    pca = PCA(n_components=n)
    pca_trained = pca.fit_transform(df)

    # Create DataFrame
    pca_df = pd.DataFrame(data=pca_trained, columns=[f'{prefix}{i+1}' for i in range(n)])

    return pca_df, pca.explained_variance_ratio_

# ---

def plot_dendrogram(df, method_list=["average", "complete", "ward", "single"], size=(20, 8)):
    """
    Plots dendrograms using different linkage methods.

    This function generates a subplot for each specified linkage method to visualize the hierarchical clustering of the given DataFrame.

    Parameters
    ----------
    - df (pd.DataFrame): The DataFrame containing the data for clustering. Rows represent samples.
    - method_list (list of str, optional): A list of linkage methods to use for generating dendrograms. Defaults to ["average", "complete", "ward", "single"].
    - size (tuple, optional): The size of the entire figure. Defaults to (20, 8).

    Returns
    -------
    - None: The function displays the dendrograms but does not return any value.
    """
    _, axes = plt.subplots(nrows=1, ncols=len(method_list), figsize=size)
    axes = axes.flat

    for i, method in enumerate(method_list):

        sch.dendrogram(sch.linkage(df, method=method),
                        labels=df.index,
                        leaf_rotation=90, leaf_font_size=10,
                        ax=axes[i])
        
        axes[i].set_title(f'Dendrogram using {method}')
        axes[i].set_xlabel('Samples')
        axes[i].set_ylabel('Distance')


def evaluate_balance(cardinality):
    """
    Evaluate the balance of cluster sizes.

    This function computes the ratio of the largest cluster size to the smallest cluster size 
    from a given dictionary of cluster sizes. It is used to assess how evenly distributed 
    the clusters are. The closer the result is to 1, the more balanced the clusters. 

    If there is only one cluster or no clusters, the function returns `float('inf')` to 
    avoid penalizing such cases. Similarly, if any cluster has a size of zero, it also 
    returns `float('inf')` to avoid division by zero.

    Parameters
    ----------
    cardinality : dict
        A dictionary where keys represent cluster identifiers and values represent 
        the sizes of the clusters.

    Returns
    -------
    float
        The balance ratio defined as `max(cluster_sizes) / min(cluster_sizes)`. 
        Returns `float('inf')` if there are fewer than two clusters or if any cluster size is zero.
    """

    cluster_sizes = list(cardinality.values())

    # Avoid penalizing only one cluster
    if len(cluster_sizes) < 2:
        return float('inf')
    
    return max(cluster_sizes) / min(cluster_sizes) if min(cluster_sizes) > 0 else float('inf')


def agglomerative_methods(df, n_min=2, n_max=5, linkage_methods = ['complete',  'ward'], distance_metrics = ['euclidean', 'cosine', 'chebyshev']):
    """
    Performs Agglomerative Clustering using various linkage methods, distance metrics, and cluster counts, and evaluates the results.

    The function iteratively applies Agglomerative Clustering with specified configurations and computes performance metrics such as Silhouette Score and Davies-Bouldin Index for each clustering result. Results are returned as a DataFrame.

    Parameters
    ----------
    - df (pd.DataFrame): The input data for clustering, where rows represent samples and columns represent features.
    - n_min (int, optional): The minimum number of clusters to evaluate. Defaults to 2.
    - n_max (int, optional): The maximum number of clusters to evaluate. Defaults to 5.
    - linkage_methods (list of str, optional): A list of linkage methods for clustering. Defaults to ['complete', 'ward'].
    - distance_metrics (list of str, optional): A list of distance metrics to use. Defaults to ['euclidean', 'cosine', 'chebyshev'].

    Returns
    -------
    - pd.DataFrame: A DataFrame containing the results with columns for linkage method, metric, silhouette score, Davies-Bouldin Index, cluster cardinality, and number of clusters.
    """

    # Results storage
    results = []

    for linkage_method in linkage_methods:
        for metric in distance_metrics:
            for cluster in range(n_min,n_max+1):

                try:
                    # Config AgglomerativeClustering model
                    modelo = AgglomerativeClustering(
                        linkage=linkage_method,
                        metric=metric,  
                        distance_threshold=None,  # We use n_clusters
                        n_clusters=cluster,
                    )
                    
                    # Model fit
                    labels = modelo.fit_predict(df)

                    # Initialize metrics
                    silhouette_avg, db_score = None, None

                    # Get metrics if we have more than one cluster
                    if len(np.unique(labels)) > 1:

                        # Silhouette Score
                        silhouette_avg = silhouette_score(df, labels, metric=metric)

                        # Davies-Bouldin Index
                        db_score = davies_bouldin_score(df, labels)

                        # Cardinality
                        cluster_cardinality = {cluster: sum(labels == cluster) for cluster in np.unique(labels)}

                    # If only one cluster
                    else:
                        cluster_cardinality = {'Unique cluster': len(df)}

                    # Store results
                    results.append({
                        'linkage': linkage_method,
                        'metric': metric,
                        'silhouette_score': silhouette_avg,
                        'davies_bouldin_index': db_score,
                        'cluster_cardinality': cluster_cardinality,
                        'n_cluster': cluster
                    })

                except Exception as e:
                    print(f"Error with linkage={linkage_method}, metric={metric}: {e}")

    # Build results dataframe
    results_df = pd.DataFrame(results)
    results_df['balance_score'] = results_df['cluster_cardinality'].apply(evaluate_balance)

    # Ranking based in all metrics
    results_df['ranking_score'] = (
        results_df['silhouette_score'] -  # Max
        results_df['davies_bouldin_index'] -  # Min
        results_df['balance_score']  # Min
    )

    return results_df


def dbscan_methods(df, eps_values=[1, 2, 3, 4, 5], min_samples_values=[5, 10, 15, 20]):
    """
    Applies the DBSCAN clustering algorithm to a given dataset for a range of `eps` and `min_samples` parameter values, evaluating clustering performance using silhouette score and Davies-Bouldin index.

    Parameters
    ----------
    - df (pd.DataFrame): The input dataset to cluster.
    - eps_values (list of float, optional): A list of epsilon values for DBSCAN. Default is [1, 2, 3, 4, 5].
    - min_samples_values (list of int, optional): A list of minimum samples values for DBSCAN. Default is [5, 10, 15, 20].

    Returns
    -------
    - (pd.DataFrame): A dataframe containing the results of clustering for each combination of `eps` and `min_samples`, including silhouette score, Davies-Bouldin index, and cluster cardinality.
    """

    # Results storage
    results = []

    for eps in eps_values:
        for min_samples in min_samples_values:

            # Apply DBSCAN
            dbscan = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1)

            # Model fit
            labels = dbscan.fit_predict(df)

            # Initialize metrics
            silhouette, db_score = None, None

            # Compute metrics ignoring noise
            if len(set(labels)) > 1 and len(set(labels)) < len(labels):

                # Silhouette Score
                silhouette = silhouette_score(df, labels)

                # Davies-Bouldin Index
                db_score = davies_bouldin_score(df, labels)

                # Cardinality
                cluster_cardinality = {cluster: sum(labels == cluster) for cluster in np.unique(labels)}

            else:
                silhouette = -1
                cluster_cardinality = {'Unique cluster': len(df)}

            # Save results
            results.append({
                "eps": eps,
                "min_samples": min_samples,
                "silhouette_score": silhouette,
                "davies_bouldin_score": db_score,
                "cardinality": cluster_cardinality
            })

    # Return metrics
    return pd.DataFrame(results).sort_values(by="silhouette_score", ascending=False)