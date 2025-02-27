import config.paths as paths
from shared.data.classes.data_aggregator import DataAggregator
from src.utils.generate_artifacts import generate_artifacts
from dataclasses import dataclass
from typing import Dict, Any
import pandas as pd


@dataclass
class DataAggregationProcessor:
    general_config: Dict[str, Any]
    data_config: Dict[str, Any]

    def run(self, data: pd.DataFrame) -> pd.DataFrame:
        data_aggregator = DataAggregator(
            data=data,
            timestamp=self.data_config.timestamp,
            aggregation_types=self.data_config.rolling_aggregation.aggregation_types,
            window=self.data_config.rolling_aggregation.window,
            min_periods=self.data_config.rolling_aggregation.min_periods,
            window_selection_frequency=self.data_config.rolling_aggregation.window_selection_frequency,
        )
        data = data_aggregator.rolling_aggregate()
        self.generate_artifacts_for_aggregate_data(data)
        return data

    def generate_artifacts_for_aggregate_data(self, data: pd.DataFrame) -> None:
        paths_dict = {
            "time_series_plots": paths.Paths.TIME_SERIES_PLOTS_FOR_AGGREGATED_FEATURES_PATH.value,
            "histogram_plots": paths.Paths.HISTOGRAM_PLOTS_FOR_AGGREGATED_FEATURES_PATH.value,
            "custom_plots": paths.Paths.CUSTOM_PLOTS_FOR_AGGREGATED_FEATURES_PATH.value,
        }
        generate_artifacts(
            self.general_config,
            self.data_config,
            data,
            "stage_7_rolling_aggregate_data",
            paths_dict,
        )
