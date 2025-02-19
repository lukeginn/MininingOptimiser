import config.paths as paths
from shared.data.export_data import export_data
from shared.data.generate_meta_data import generate_meta_data
from shared.data.generate_time_series_plots import generate_time_series_plots
from shared.data.generate_histogram_plots import generate_histogram_plots
from src.visualisation.custom_plots import custom_plots
from shared.data.introduce_lags import introduce_lags

def introducing_lags(data, config):
    if config.data.introduce_lags.run:
        data = introduce_lags(
            data=data,
            timestamp=config.data.timestamp,
            features=config.data.introduce_lags.features,
            lags=config.data.introduce_lags.lags,
            optimise_lags=config.data.introduce_lags.optimise_lags,
            target=config.data.introduce_lags.target,
            max_lag=config.data.introduce_lags.max_lag,
            overwrite_existing_features=config.data.introduce_lags.overwrite_existing_features,
        )
        if config.export_data:
            export_data(
                data=data,
                path=paths.Paths.EXPORTED_DATA_FILE.value,
                path_suffix="stage_6_lags_introduced",
            )
        if config.generate_meta_data:
            generate_meta_data(
                data=data,
                path=paths.Paths.META_DATA_FILE.value,
                path_suffix="stage_6_lags_introduced",
            )
        if config.generate_time_series_plots:
            generate_time_series_plots(
                data=data,
                timestamp=config.data.timestamp,
                path=paths.Paths.TIME_SERIES_PLOTS_FOR_LAGGED_FEATURES_PATH.value,
            )
        if config.generate_histogram_plots.run:
            generate_histogram_plots(
                data=data,
                path=paths.Paths.HISTOGRAM_PLOTS_FOR_LAGGED_FEATURES_PATH.value,
                number_of_bins=config.generate_histogram_plots.number_of_bins,
            )
        if config.generate_custom_plots:
            custom_plots(
                data=data,
                path=paths.Paths.CUSTOM_PLOTS_FOR_LAGGED_FEATURES_PATH.value
            )
    return data