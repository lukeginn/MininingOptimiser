import pandas as pd
import logging as logger
from dataclasses import dataclass, field
from typing import List, Optional, Union

@dataclass
class DataAggregator:
    data: pd.DataFrame
    timestamp: str
    features_to_aggregate: Optional[List[str]] = field(default_factory=list)
    aggregation_types: List[str] = field(default_factory=lambda: ["mean", "std", "min", "max"])
    window: int = 144
    min_periods: int = 144
    window_selection_frequency: int = 144

    def aggregate(self) -> pd.DataFrame:
        self.data = self._format_timestamp(freq="D")

        if not self.features_to_aggregate:
            self.features_to_aggregate = self._identify_features_to_aggregate()

        self.data = self._aggregate_using_timestamp()

        return self.data

    def rolling_aggregate(self) -> pd.DataFrame:
        self.data = self._sort_data_by_timestamp()

        if not self.features_to_aggregate:
            self.features_to_aggregate = self._identify_features_to_aggregate()

        self.data = self._rolling_aggregate_using_timestamp()

        self.data = self._select_every_nth_row()

        return self.data

    def _identify_features_to_aggregate(self) -> List[str]:
        # Identify the features to aggregate
        return self.data.select_dtypes(include="number").columns.tolist()

    def _aggregate_using_timestamp(self) -> pd.DataFrame:
        if not isinstance(self.aggregation_types, list):
            self.aggregation_types = [self.aggregation_types]

        # Aggregate the data
        aggregated_data = self.data.groupby(self.timestamp).agg(
            {feature: self.aggregation_types for feature in self.features_to_aggregate}
        )

        # Flatten the column names
        aggregated_data.columns = ["_".join(x) for x in aggregated_data.columns.to_flat_index()]

        return aggregated_data

    def _format_timestamp(self, freq: str = "D") -> pd.DataFrame:
        # Format the timestamp into the specified frequency
        self.data[self.timestamp] = self.data[self.timestamp].dt.floor(freq)
        return self.data

    def _rolling_aggregate_using_timestamp(self) -> pd.DataFrame:
        if not isinstance(self.aggregation_types, list):
            self.aggregation_types = [self.aggregation_types]

        # Apply rolling aggregation
        rolling_data = (
            self.data[self.features_to_aggregate]
            .rolling(window=self.window, min_periods=self.min_periods)
            .agg(self.aggregation_types)
        )

        # Flatten the column names
        rolling_data.columns = ["_".join(x) for x in rolling_data.columns.to_flat_index()]

        # Combine the rolling data with the original timestamp
        combined_data = pd.concat([self.data[self.timestamp], rolling_data], axis=1).reset_index(drop=True)

        # Filter out the first few rows with NaN values from min_periods
        combined_data = combined_data.iloc[self.min_periods:]

        return combined_data

    def _sort_data_by_timestamp(self) -> pd.DataFrame:
        self.data = self.data.sort_values(by=self.timestamp).reset_index(drop=True)
        return self.data

    def _select_every_nth_row(self) -> pd.DataFrame:
        # Ensure n is an integer and at least 1
        n = max(1, int(round(self.window_selection_frequency)))
        self.data = self.data.iloc[::n]
        return self.data
