# Data processing
# -----------------------------------------------------------------------
import numpy as np
import pandas as pd
import math

# For visualizations
# -----------------------------------------------------------------------
import matplotlib.pyplot as plt
import seaborn as sns

# Extracting the number of clusters and metrics
# -----------------------------------------------------------------------
from yellowbrick.cluster import KElbowVisualizer
from sklearn.metrics import silhouette_score, davies_bouldin_score

# Clustering models
# -----------------------------------------------------------------------
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN
from sklearn.cluster import SpectralClustering

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


def plot_combined_target_distribution(df, target, feature, size=(10, 6)):
    """
    Plots the combined distribution of a feature and the proportion of a binary target variable.

    This function creates a dual-axis histogram visualization. The primary axis shows the distribution of the specified feature, while the secondary axis overlays the proportion of the target variable.

    Parameters:
    - df (pd.DataFrame): The DataFrame containing the data to plot.
    - target (str): The name of the binary target column in the DataFrame.
    - feature (str): The name of the feature column to plot.
    - figsize (tuple, optional): Figure size for the plot. Defaults to (10, 6).

    Returns:
    - None: The function displays the plot but does not return any value.
    """

    fig, ax = plt.subplots(figsize=size)

    fig.suptitle(f"Proportion of '{target}' by '{feature}' distribution")
    
    # Total count histogram
    sns.histplot(data=df,
                x=feature,
                bins='auto',
                ax=ax)

    # Second histogram for positive class probability
    ax2 = ax.twinx()

    sns.histplot(data=df,
             x=feature,
             hue=target,  
             stat="probability", # Normalize to show proportion
             bins='auto',
             multiple="fill", # Get maximum in the top
             palette={1: "red", 0: "#FFFFFF"}, # Red for possitive, white for negative
             ax=ax2,
             alpha=0.25, # Transparency, important here
             edgecolor=None)

    # Set ylim for proportion
    ax2.set_ylim(0, 1)

    # Remove ax2 legend, we only want one
    ax2.get_legend().remove()
    fig.legend([f"{feature.capitalize()} distribution", f"{target.capitalize()} proportion"], loc="upper right")

    plt.tight_layout()
    plt.show()


def plot_radar(df, columns):
    """
    Plots a radar chart to visualize the mean values of specified columns for each cluster.

    The function calculates the mean values of the provided columns grouped by the 'cluster' column and creates a radar chart to compare these means across clusters.

    Parameters
    ----------
        - df (pd.DataFrame): The DataFrame containing the data, including a 'cluster' column.
        - columns (list of str): A list of column names to include in the radar plot.

    Returns
    -------
        - None: The function displays the radar plot but does not return any value.
    """

    # Group by cluster and compute mean
    cluster_means = df.groupby('cluster')[columns].mean()

    # Repeat first column at the end to close radar
    cluster_means = pd.concat([cluster_means, cluster_means.iloc[:, 0:1]], axis=1)

    # Angles for radarplot
    num_vars = len(columns)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]  # Close radar

    _, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    # A radar for each cluster
    for i, row in cluster_means.iterrows():
        ax.plot(angles, row, label=f'Cluster {i}')
        ax.fill(angles, row, alpha=0.25)

    # Axis tags
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(columns)

    # Legend and title
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    
    plt.tight_layout()
    plt.show()


def plot_clusters(df):
    """
    Plots bar charts showing the mean values of each feature grouped by clusters.

    The function iterates through the features in the DataFrame (excluding the 'cluster' column) and creates a subplot for each, displaying the mean value of the feature per cluster.

    Parameters
    ----------
        - df (pd.DataFrame): The DataFrame containing the data, including a 'cluster' column and other feature columns.

    Returns
    -------
        - None: The function displays the bar chart plots but does not return any value.
    """

    cols = df.drop(columns='cluster').columns.to_list()

    fig, axes = plt.subplots(nrows=2, ncols=math.ceil(len(cols)/2), figsize=(20, 8))
    axes = axes.flat

    for i, col in enumerate(cols):

        df_group = df.groupby('cluster')[col].mean().reset_index()

        sns.barplot(x='cluster', y=col, data=df_group, ax=axes[i], palette='coolwarm')
        axes[i].set_title(col, fontsize=18)

    if len(cols) % 2 != 0:
        fig.delaxes(axes[-1])

    plt.tight_layout()


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