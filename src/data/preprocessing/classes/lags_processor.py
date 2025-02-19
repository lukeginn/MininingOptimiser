import config.paths as paths
from shared.data.introduce_lags import introduce_lags
from src.utils.generate_artifacts import generate_artifacts

class LagsProcessor:
    def __init__(self, config):
        self.config = config

    def introduce_lags(self, data):
        if self.config.data.introduce_lags.run:
            data = introduce_lags(
                data=data,
                timestamp=self.config.data.timestamp,
                features=self.config.data.introduce_lags.features,
                lags=self.config.data.introduce_lags.lags,
                optimise_lags=self.config.data.introduce_lags.optimise_lags,
                target=self.config.data.introduce_lags.target,
                max_lag=self.config.data.introduce_lags.max_lag,
                overwrite_existing_features=self.config.data.introduce_lags.overwrite_existing_features,
            )
            self.generate_artifacts_for_introduce_lags(data)
        return data
    
    def generate_artifacts_for_introduce_lags(self, data):
        paths_dict = {
            'time_series_plots': paths.Paths.TIME_SERIES_PLOTS_FOR_LAGGED_FEATURES_PATH.value,
            'histogram_plots': paths.Paths.HISTOGRAM_PLOTS_FOR_LAGGED_FEATURES_PATH.value,
            'custom_plots': paths.Paths.CUSTOM_PLOTS_FOR_LAGGED_FEATURES_PATH.value
        }
        generate_artifacts(self.config, data, "stage_6_lags_introduced", paths_dict)