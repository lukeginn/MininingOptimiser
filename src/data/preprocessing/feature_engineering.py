import logging as logger
import config.paths as paths
from shared.data.export_data import export_data
from shared.data.generate_meta_data import generate_meta_data
from shared.data.generate_time_series_plots import generate_time_series_plots
from shared.data.generate_histogram_plots import generate_histogram_plots
from src.visualisation.custom_plots import custom_plots

def run_feature_engineering(data, config):
    if config.data.feature_engineering.run:
        data = feature_engineering(
            data=data
        )
        if config.export_data:
            export_data(
                data=data,
                path=paths.Paths.EXPORTED_DATA_FILE.value,
                path_suffix="stage_8_feature_engineered_data",
            )
        if config.generate_meta_data:
            generate_meta_data(
                data=data,
                path=paths.Paths.META_DATA_FILE.value,
                path_suffix="stage_8_feature_engineered_data",
            )
        if config.generate_time_series_plots:
            generate_time_series_plots(
                data=data,
                timestamp=config.data.timestamp,
                path=paths.Paths.TIME_SERIES_PLOT_FOR_FEATURE_ENGINEERED_DATA_PATH.value,
            )
        if config.generate_histogram_plots.run:
            generate_histogram_plots(
                data=data,
                path=paths.Paths.HISTOGRAM_PLOTS_FOR_FEATURE_ENGINEERED_DATA_PATH.value,
                number_of_bins=config.generate_histogram_plots.number_of_bins,
            )
        if config.generate_custom_plots:
            custom_plots(
                data=data,
                path=paths.Paths.CUSTOM_PLOTS_FOR_FEATURE_ENGINEERED_DATA_PATH.value
            )
    return data

def feature_engineering(data):
    logger.info("Feature engineering started")

    # No feature engineering currently occuring

    logger.info("Feature engineering completed successfully")   

    return data