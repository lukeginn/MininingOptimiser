import logging as logger
from sklearn.cluster import DBSCAN
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from dataclasses import dataclass, field
from typing import List, Optional
import pandas as pd


@dataclass
class OutlierIdentifier:
    data: pd.DataFrame
    features: Optional[List[str]] = field(default_factory=list)
    method: str = "mad"
    iqr_threshold: float = 1.5
    z_score_threshold: float = 3
    mad_threshold: float = 3
    dbscan_eps: float = 0.5
    dbscan_min_samples: int = 5
    isolation_forest_threshold: float = 0.001
    lof_threshold: float = 0.001

    def run(self) -> pd.DataFrame:
        logger.info("Starting outlier identification process.")
        logger.info("Method to be used: %s", self.method)

        if not self.features:
            self.features = self._numerical_features()

        for feature in self.features:
            logger.info(f"Processing feature: {feature}")
            if self.method == "iqr":
                self.data = self._identify_outliers_via_iqr(feature)
            elif self.method == "z_score":
                self.data = self._identify_outliers_via_z_score(feature)
            elif self.method == "mad":
                self.data = self._identify_outliers_via_mad(feature)
            elif self.method == "dbscan":
                self.data = self._identify_outliers_via_dbscan(feature)
            elif self.method == "isolation_forest":
                self.data = self._identify_outliers_via_isolation_forest(feature)
            elif self.method == "lof":
                self.data = self._identify_outliers_via_lof(feature)

        return self.data

    def _numerical_features(self) -> List[str]:
        return self.data.select_dtypes(include="number").columns.tolist()

    def _identify_outliers_via_iqr(self, feature: str) -> pd.DataFrame:
        Q1 = self.data[feature].quantile(0.25)
        Q3 = self.data[feature].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - self.iqr_threshold * IQR
        upper_bound = Q3 + self.iqr_threshold * IQR
        self.data[feature] = self.data[feature].mask(
            (self.data[feature] < lower_bound) | (self.data[feature] > upper_bound)
        )
        return self.data

    def _identify_outliers_via_z_score(self, feature: str) -> pd.DataFrame:
        z_scores = (self.data[feature] - self.data[feature].mean()) / self.data[
            feature
        ].std()
        self.data[feature] = self.data[feature].mask(
            abs(z_scores) > self.z_score_threshold
        )
        return self.data

    def _identify_outliers_via_mad(self, feature: str) -> pd.DataFrame:
        if self.data[feature].nunique() < 2:
            logger.warning(
                f"Feature {feature} has less than 2 unique values. Skipping MAD outlier detection."
            )
            return self.data
        median = self.data[feature].median()
        mad = (self.data[feature] - median).abs().median()
        if mad == 0:
            logger.warning(
                f"Feature {feature} has a MAD of 0. Skipping MAD outlier detection."
            )
            return self.data
        self.data[feature] = self.data[feature].mask(
            abs(self.data[feature] - median) / mad > self.mad_threshold
        )
        return self.data

    def _identify_outliers_via_dbscan(self, feature: str) -> pd.DataFrame:
        dbscan = DBSCAN(eps=self.dbscan_eps, min_samples=self.dbscan_min_samples)
        data_copy = self.data.ffill().bfill()
        outliers = dbscan.fit_predict(data_copy[feature].values.reshape(-1, 1))
        self.data[feature] = self.data[feature].mask(outliers == -1)
        return self.data

    def _identify_outliers_via_isolation_forest(self, feature: str) -> pd.DataFrame:
        if not (0.0 < self.isolation_forest_threshold <= 0.5):
            logger.warning(
                f"Threshold for IsolationForest must be in the range (0.0, 0.5]. Got {self.isolation_forest_threshold}. Setting to default 0.05."
            )
            self.isolation_forest_threshold = 0.001
        isolation_forest = IsolationForest(
            contamination=self.isolation_forest_threshold
        )
        data_copy = self.data.ffill().bfill()
        outliers = isolation_forest.fit_predict(
            data_copy[feature].values.reshape(-1, 1)
        )
        self.data[feature] = self.data[feature].mask(outliers == -1)
        return self.data

    def _identify_outliers_via_lof(self, feature: str) -> pd.DataFrame:
        if not (0.0 < self.lof_threshold <= 0.5):
            logger.warning(
                f"Threshold for LOF must be in the range (0.0, 0.5]. Got {self.lof_threshold}. Setting to default 0.05."
            )
            self.lof_threshold = 0.001
        lof = LocalOutlierFactor(n_neighbors=20, contamination=self.lof_threshold)
        data_copy = self.data.ffill().bfill()
        outliers = lof.fit_predict(data_copy[feature].values.reshape(-1, 1))
        self.data[feature] = self.data[feature].mask(outliers == -1)
        return self.data
