import config.paths as paths
from shared.data.aggregate_data import rolling_aggregate_data_via_timestamp
from src.utils.generate_artifacts import generate_artifacts

class DataAggregator:
    def __init__(self, config):
        self.config = config

    def aggregate_data(self, data):
        data = rolling_aggregate_data_via_timestamp(
            data=data,
            timestamp=self.config.data.timestamp,
            aggregation_types=self.config.data.rolling_aggregation.aggregation_types,
            window=self.config.data.rolling_aggregation.window,
            min_periods=self.config.data.rolling_aggregation.min_periods,
            window_selection_frequency=self.config.data.rolling_aggregation.window_selection_frequency,
        )
        self.generate_artifacts_for_aggregate_data(data)
        return data
    
    def generate_artifacts_for_aggregate_data(self, data):
        paths_dict = {
            'time_series_plots': paths.Paths.TIME_SERIES_PLOTS_FOR_AGGREGATED_FEATURES_PATH.value,
            'histogram_plots': paths.Paths.HISTOGRAM_PLOTS_FOR_AGGREGATED_FEATURES_PATH.value,
            'custom_plots': paths.Paths.CUSTOM_PLOTS_FOR_AGGREGATED_FEATURES_PATH.value
        }
        generate_artifacts(self.config, data, "stage_7_rolling_aggregate_data", paths_dict)