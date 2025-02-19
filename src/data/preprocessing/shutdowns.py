import config.paths as paths
from shared.data.export_data import export_data
from shared.data.generate_meta_data import generate_meta_data
from shared.data.generate_time_series_plots import generate_time_series_plots
from shared.data.generate_histogram_plots import generate_histogram_plots
from src.data.custom_plots import custom_plots

def run_filter_shutdowns(data, config):
    if config.data.filter_shutdowns.run:
        data = filter_shutdowns(data)
        if config.export_data:
            export_data(
                data=data,
                path=paths.Paths.EXPORTED_DATA_FILE.value,
                path_suffix="stage_10_filter_shutdowns",
            )
        if config.generate_meta_data:
            generate_meta_data(
                data=data,
                path=paths.Paths.META_DATA_FILE.value,
                path_suffix="stage_10_filter_shutdowns",
            )
        if config.generate_time_series_plots:
            generate_time_series_plots(
                data=data,
                timestamp=config.data.timestamp,
                path=paths.Paths.TIME_SERIES_PLOTS_FOR_FILTERING_SHUTDOWN_PATH.value,
            )
        if config.generate_histogram_plots.run:
            generate_histogram_plots(
                data=data,
                path=paths.Paths.HISTOGRAM_PLOTS_FOR_FILTERING_SHUTDOWN_PATH.value,
                number_of_bins=config.generate_histogram_plots.number_of_bins,
            )
        if config.generate_custom_plots:
            custom_plots(
                data=data,
                path=paths.Paths.CUSTOM_PLOTS_FOR_FILTERING_SHUTDOWN_PATH.value
            )
    return data

def filter_shutdowns(data):
    shutdown_dates = identify_shutdown_times(data)
    if shutdown_dates is not None:
        data = data[~data['DATE'].isin(shutdown_dates)]
    return data


def identify_shutdown_times(data):
    
    # Currently no shutdown periods have been identified
    shutdown_dates = None

    return shutdown_dates