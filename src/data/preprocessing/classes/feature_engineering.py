import logging as logger
import config.paths as paths
from src.utils.generate_artifacts import generate_artifacts


class FeatureEngineering:
    def __init__(self, general_config, data_config):
        self.general_config = general_config
        self.data_config = data_config

    def run(self, data):
        if self.data_config.feature_engineering.run:
            data = self.feature_engineering(data)
            self.generate_artifacts_for_run(data)
        return data

    def generate_artifacts_for_run(self, data):
        paths_dict = {
            "time_series_plots": paths.Paths.TIME_SERIES_PLOT_FOR_FEATURE_ENGINEERED_DATA_PATH.value,
            "histogram_plots": paths.Paths.HISTOGRAM_PLOTS_FOR_FEATURE_ENGINEERED_DATA_PATH.value,
            "custom_plots": paths.Paths.CUSTOM_PLOTS_FOR_FEATURE_ENGINEERED_DATA_PATH.value,
        }
        generate_artifacts(
            self.general_config, data, "stage_8_feature_engineered_data", paths_dict
        )

    def feature_engineering(self, data):
        logger.info("Feature engineering started")

        # No feature engineering currently occurring

        logger.info("Feature engineering completed successfully")

        return data
