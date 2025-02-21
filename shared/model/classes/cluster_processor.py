import numpy as np
import pandas as pd
import logging as logger
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from typing import List, Optional, Union
from dataclasses import dataclass, field


@dataclass
class ClusterProcessor:
    data: pd.DataFrame
    training_features: List[str]
    informational_features: Optional[List[str]] = None
    include_row_count_sum: bool = True
    path: Optional[str] = None
    model_choice: str = "kmeans"
    k_means_n_clusters: int = 3
    k_means_max_iter: int = 300
    dbscan_eps: float = 0.5
    dbscan_min_samples: int = 5
    agglomerative_n_clusters: int = 3
    random_state: int = 1

    def run(self) -> pd.DataFrame:
        clusters = self._get_clusters()
        self.data["cluster"] = clusters

        cluster_centers = self._get_cluster_centers()
        cluster_centers = self._apply_suffixes(cluster_centers)
        cluster_centers = self._reorder_columns(cluster_centers)

        if self.path:
            self._export_cluster_centers(cluster_centers)

        return cluster_centers

    def _get_clusters(self) -> np.ndarray:
        logger.info(f"Running clustering with model_choice: {self.model_choice}")
        logger.info(
            f"Generating clusters with the following training features: {self.training_features}"
        )

        X = self.data[self.training_features]

        np.random.seed(self.random_state)
        if self.model_choice == "kmeans":
            model = KMeans(
                n_clusters=self.k_means_n_clusters, max_iter=self.k_means_max_iter
            )
        elif self.model_choice == "dbscan":
            model = DBSCAN(eps=self.dbscan_eps, min_samples=self.dbscan_min_samples)
        elif self.model_choice == "agglomerative":
            model = AgglomerativeClustering(n_clusters=self.agglomerative_n_clusters)
        else:
            raise ValueError(f"Unsupported model_choice: {self.model_choice}")

        clusters = model.fit_predict(X)
        return clusters

    def _get_cluster_centers(self) -> pd.DataFrame:
        if self.informational_features:
            logger.info(
                f"Calculating cluster centers with the following informational features: {self.informational_features}"
            )
            features = self.training_features + self.informational_features
        else:
            features = self.training_features

        cluster_centers = self.data.groupby("cluster")[features].mean()

        if self.include_row_count_sum:
            cluster_centers["row_count"] = self.data.groupby("cluster").size()
            cluster_centers["row_count_proportion"] = cluster_centers[
                "row_count"
            ] / len(self.data)

        return cluster_centers

    def _apply_suffixes(self, data: pd.DataFrame) -> pd.DataFrame:
        columns_to_suffix = [
            col
            for col in data.columns
            if col not in ["row_count", "row_count_proportion"]
        ]
        data.rename(
            columns={col: f"{col}_historical_actuals" for col in columns_to_suffix},
            inplace=True,
        )
        return data

    def _reorder_columns(self, data: pd.DataFrame) -> pd.DataFrame:
        columns = data.columns.tolist()
        columns.remove("row_count")
        columns.remove("row_count_proportion")
        columns = ["row_count", "row_count_proportion"] + columns
        data = data[columns]
        return data

    def _export_cluster_centers(self, cluster_centers: pd.DataFrame) -> None:
        cluster_centers.to_csv(self.path)
        logger.info(f"Cluster centers exported to {self.path}")
