import config.paths as paths
from shared.data.export_data import export_data
from shared.data.generate_meta_data import generate_meta_data
from shared.data.generate_time_series_plots import generate_time_series_plots
from shared.data.generate_histogram_plots import generate_histogram_plots
from src.visualisation.custom_plots import custom_plots
from shared.data.aggregate_data import rolling_aggregate_data_via_timestamp

def aggregating_data(data, config):
    data = rolling_aggregate_data_via_timestamp(
        data=data,
        timestamp=config.data.timestamp,
        aggregation_types=config.data.rolling_aggregation.aggregation_types,
        window=config.data.rolling_aggregation.window,
        min_periods=config.data.rolling_aggregation.min_periods,
        window_selection_frequency=config.data.rolling_aggregation.window_selection_frequency,
    )
    if config.export_data:
        export_data(
            data=data,
            path=paths.Paths.EXPORTED_DATA_FILE.value,
            path_suffix="stage_7_rolling_aggregate_data",
        )
    if config.generate_meta_data:
        generate_meta_data(
            data=data,
            path=paths.Paths.META_DATA_FILE.value,
            path_suffix="stage_7_rolling_aggregate_data",
        )
    if config.generate_time_series_plots:
        generate_time_series_plots(
            data=data,
            timestamp=config.data.timestamp,
            path=paths.Paths.TIME_SERIES_PLOTS_FOR_AGGREGATED_FEATURES_PATH.value,
        )
    if config.generate_histogram_plots.run:
        generate_histogram_plots(
            data=data,
            path=paths.Paths.HISTOGRAM_PLOTS_FOR_AGGREGATED_FEATURES_PATH.value,
            number_of_bins=config.generate_histogram_plots.number_of_bins,
        )
    if config.generate_custom_plots:
        custom_plots(
            data=data,
            path=paths.Paths.CUSTOM_PLOTS_FOR_AGGREGATED_FEATURES_PATH.value
        )
    return data