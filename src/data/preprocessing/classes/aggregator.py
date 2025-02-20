import config.paths as paths
from shared.data.aggregate_data import rolling_aggregate_data_via_timestamp
from src.utils.generate_artifacts import generate_artifacts


class DataAggregator:
    def __init__(self, general_config, data_config):
        self.general_config = general_config
        self.data_config = data_config

    def run(self, data):
        data = rolling_aggregate_data_via_timestamp(
            data=data,
            timestamp=self.data_config.timestamp,
            aggregation_types=self.data_config.rolling_aggregation.aggregation_types,
            window=self.data_config.rolling_aggregation.window,
            min_periods=self.data_config.rolling_aggregation.min_periods,
            window_selection_frequency=self.data_config.rolling_aggregation.window_selection_frequency,
        )
        self.generate_artifacts_for_aggregate_data(data)
        return data

    def generate_artifacts_for_aggregate_data(self, data):
        paths_dict = {
            "time_series_plots": paths.Paths.TIME_SERIES_PLOTS_FOR_AGGREGATED_FEATURES_PATH.value,
            "histogram_plots": paths.Paths.HISTOGRAM_PLOTS_FOR_AGGREGATED_FEATURES_PATH.value,
            "custom_plots": paths.Paths.CUSTOM_PLOTS_FOR_AGGREGATED_FEATURES_PATH.value,
        }
        generate_artifacts(
            self.general_config, data, "stage_7_rolling_aggregate_data", paths_dict
        )
