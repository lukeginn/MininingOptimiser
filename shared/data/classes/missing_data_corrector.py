import logging as logger
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class MissingDataCorrector:
    data: pd.DataFrame
    delete_all_rows_with_missing_values: bool = False
    interpolate_time_series: bool = False
    interpolate_highly_regular_time_series: bool = False
    replace_missing_values_with_x: bool = True
    replace_missing_values_with_last_known_value: bool = False
    interpolate_method: str = "linear"
    interpolate_limit_direction: str = "both"
    interpolate_max_gap: int = 5
    interpolate_highly_regular_method: str = "linear"
    interpolate_highly_regular_limit_direction: str = "both"
    interpolate_highly_regular_interval_min: int = 144
    replace_missing_values_with_last_known_value_backfill: bool = False
    timestamp: Optional[str] = None
    features: Optional[List[str]] = field(default_factory=list)
    x: int = 0

    def run(self) -> pd.DataFrame:
        logger.info("Correcting missing data")
        if not self.features:
            self.features = self._numerical_features()
            self.features = self._features_with_variance()

        if self.interpolate_time_series:
            self.data = self._interpolate_time_series_features()

        if self.interpolate_highly_regular_time_series:
            self.data = self._interpolate_highly_regular_time_series_features()

        if (
            self.replace_missing_values_with_x
            or self.replace_missing_values_with_last_known_value
        ):
            self.data = self._replace_missing_values()

        if self.delete_all_rows_with_missing_values:
            self.data = self._delete_rows_with_missing_values()

        return self.data

    def _numerical_features(self) -> List[str]:
        return self.data.select_dtypes(include="number").columns.tolist()

    def _features_with_variance(self) -> List[str]:
        # Select features that have at least one non-missing value
        return self.data.columns[self.data.notna().any()].tolist()

    def _delete_rows_with_missing_values(self) -> pd.DataFrame:
        logger.info("Deleting rows with missing values")
        return self.data.dropna(subset=self.features)

    def _replace_missing_values(self) -> pd.DataFrame:
        if self.replace_missing_values_with_last_known_value:
            logger.info("Replacing missing values with last known value per feature")
            for feature in self.features:
                self.data[feature] = self.data[feature].fillna(method="ffill")
                if self.replace_missing_values_with_last_known_value_backfill:
                    self.data[feature] = self.data[feature].fillna(method="bfill")
        else:
            logger.info(f"Replacing missing values with {self.x}")
            for feature in self.features:
                self.data[feature] = self.data[feature].fillna(self.x)
        return self.data

    def _interpolate_time_series_features(self) -> pd.DataFrame:
        logger.info("Interpolating time series features")
        if not self.features:
            self.features = self._numerical_features()
        self.data = self._sort_data_by_timestamp()
        self.data = self._interpolate_features()
        return self.data

    def _interpolate_highly_regular_time_series_features(self) -> pd.DataFrame:
        logger.info("Interpolating highly regular time series features")
        if not self.features:
            self.features = self._numerical_features()
        self.data = self._sort_data_by_timestamp()
        self.data = self._interpolate_highly_regular_features()
        return self.data

    def _sort_data_by_timestamp(self) -> pd.DataFrame:
        self.data = self.data.sort_values(by=self.timestamp)
        logger.info("Data sorted by timestamp")
        return self.data

    def _interpolate_features(self) -> pd.DataFrame:
        for feature in self.features:
            self.data = self._interpolate_one_feature(feature)
        logger.info("Features interpolated")
        return self.data

    def _interpolate_highly_regular_features(self) -> pd.DataFrame:
        for feature in self.features:
            self.data = self._interpolate_one_feature_with_high_and_regular_sparsity(
                feature
            )
        logger.info("Features interpolated")
        return self.data

    def _interpolate_one_feature(self, feature: str) -> pd.DataFrame:
        # Check if the feature is a datetime feature, if it is, skip it
        if np.issubdtype(self.data[feature].dtype, np.datetime64):
            logger.info(f"Skipping datetime feature: {feature}")
            return self.data

        # Identify the gaps in the data
        is_nan = self.data[feature].isna()
        gap_sizes = is_nan.groupby((is_nan != is_nan.shift()).cumsum()).transform("sum")

        # Make a copy of the original feature
        original_feature = self.data[feature].copy()

        # Interpolate the missing values
        if self.interpolate_method == "bfill":
            self.data[feature] = self.data[feature].fillna(method="bfill")
        else:
            self.data[feature] = self.data[feature].interpolate(
                method=self.interpolate_method,
                limit_direction=self.interpolate_limit_direction,
            )

        # Where the gap_sizes are greater than the max_gap, replace the values with the original values
        self.data[feature] = self.data[feature].where(
            gap_sizes <= self.interpolate_max_gap, other=original_feature
        )

        return self.data

    def _interpolate_one_feature_with_high_and_regular_sparsity(
        self, feature: str
    ) -> pd.DataFrame:
        # Check if the feature has regular intervals of non-missing values
        non_missing_indices = self.data[feature].dropna().index
        if len(non_missing_indices) > 1:
            intervals = non_missing_indices.to_series().diff().dropna()
            if (intervals >= self.interpolate_highly_regular_interval_min).mode().all():
                logger.info(
                    f"Feature {feature} has regular intervals of non-missing values"
                )

                # Interpolate the missing values
                if self.interpolate_highly_regular_method == "bfill":
                    self.data[feature] = self.data[feature].fillna(method="bfill")
                else:
                    self.data[feature] = self.data[feature].interpolate(
                        method=self.interpolate_highly_regular_method,
                        limit_direction=self.interpolate_highly_regular_limit_direction,
                    )
            else:
                logger.info(
                    f"Feature {feature} does not have regular intervals of non-missing values"
                )
        else:
            logger.info(
                f"Feature {feature} does not have enough non-missing values to determine regular intervals"
            )

        return self.data
