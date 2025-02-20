import config.paths as paths
from shared.data.introduce_lags import introduce_lags
from src.utils.generate_artifacts import generate_artifacts
from dataclasses import dataclass


@dataclass
class LagsProcessor:
    general_config: dict
    data_config: dict

    def run(self, data):
        if self.data_config.introduce_lags.run:
            data = introduce_lags(
                data=data,
                timestamp=self.data_config.timestamp,
                features=self.data_config.introduce_lags.features,
                lags=self.data_config.introduce_lags.lags,
                optimise_lags=self.data_config.introduce_lags.optimise_lags,
                target=self.data_config.introduce_lags.target,
                max_lag=self.data_config.introduce_lags.max_lag,
                overwrite_existing_features=self.data_config.introduce_lags.overwrite_existing_features,
            )
            self.generate_artifacts_for_introduce_lags(data)
        return data

    def generate_artifacts_for_introduce_lags(self, data):
        paths_dict = {
            "time_series_plots": paths.Paths.TIME_SERIES_PLOTS_FOR_LAGGED_FEATURES_PATH.value,
            "histogram_plots": paths.Paths.HISTOGRAM_PLOTS_FOR_LAGGED_FEATURES_PATH.value,
            "custom_plots": paths.Paths.CUSTOM_PLOTS_FOR_LAGGED_FEATURES_PATH.value,
        }
        generate_artifacts(
            self.general_config, data, "stage_6_lags_introduced", paths_dict
        )
