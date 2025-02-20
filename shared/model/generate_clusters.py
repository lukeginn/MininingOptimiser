import numpy as np
import pandas as pd
import logging as logger
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering


def run_clustering(
    data,
    training_features,
    informational_features=None,
    include_row_count_sum=True,
    path=None,
    model_choice="kmeans",
    k_means_n_clusters=3,
    k_means_max_iter=300,
    dbscan_eps=0.5,
    dbscan_min_samples=5,
    agglomerative_n_clusters=3,
    random_state=1,
):

    clusters = get_clusters(
        data=data,
        training_features=training_features,
        model_choice=model_choice,
        k_means_n_clusters=k_means_n_clusters,
        k_means_max_iter=k_means_max_iter,
        dbscan_eps=dbscan_eps,
        dbscan_min_samples=dbscan_min_samples,
        agglomerative_n_clusters=agglomerative_n_clusters,
        random_state=random_state,
    )
    data["cluster"] = clusters

    cluster_centers = get_cluster_centers(
        data=data,
        training_features=training_features,
        informational_features=informational_features,
        include_row_count_sum=include_row_count_sum,
    )

    cluster_centers = apply_suffixes(cluster_centers)
    cluster_centers = reorder_columns(cluster_centers)

    if path:
        export_cluster_centers(cluster_centers=cluster_centers, path=path)

    return cluster_centers


def get_clusters(
    data,
    training_features,
    model_choice,
    k_means_n_clusters,
    k_means_max_iter,
    dbscan_eps,
    dbscan_min_samples,
    agglomerative_n_clusters,
    random_state,
):
    logger.info(f"Running clustering with model_choice: {model_choice}")
    logger.info(
        f"Generating clusters with the following training features: {training_features}"
    )

    X = data[training_features]

    np.random.seed(random_state)
    if model_choice == "kmeans":
        model = KMeans(n_clusters=k_means_n_clusters, max_iter=k_means_max_iter)
    elif model_choice == "dbscan":
        model = DBSCAN(eps=dbscan_eps, min_samples=dbscan_min_samples)
    elif model_choice == "agglomerative":
        model = AgglomerativeClustering(n_clusters=agglomerative_n_clusters)
    else:
        raise ValueError(f"Unsupported model_choice: {model_choice}")

    clusters = model.fit_predict(X)

    return clusters


def get_cluster_centers(
    data, training_features, informational_features, include_row_count_sum
):
    if informational_features:
        logger.info(
            f"Calculating cluster centers with the following informational features: {informational_features}"
        )
        features = training_features + informational_features
    else:
        features = training_features

    cluster_centers = data.groupby("cluster")[features].mean()

    if include_row_count_sum:
        cluster_centers["row_count"] = data.groupby("cluster").size()
        cluster_centers["row_count_proportion"] = cluster_centers["row_count"] / len(
            data
        )

    return cluster_centers


def apply_suffixes(data):

    columns_to_suffix = [
        col for col in data.columns if col not in ["row_count", "row_count_proportion"]
    ]
    data.rename(
        columns={col: f"{col}_historical_actuals" for col in columns_to_suffix},
        inplace=True,
    )

    return data


def reorder_columns(data):
    columns = data.columns.tolist()
    columns.remove("row_count")
    columns.remove("row_count_proportion")
    columns = ["row_count", "row_count_proportion"] + columns
    data = data[columns]

    return data


def export_cluster_centers(cluster_centers, path):
    cluster_centers.to_csv(path)
    logger.info(f"Cluster centers exported to {path}")
