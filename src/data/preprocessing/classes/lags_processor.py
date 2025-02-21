import config.paths as paths
from shared.data.classes.lag_introducer import LagIntroducer
from src.utils.generate_artifacts import generate_artifacts
from dataclasses import dataclass
from typing import Dict, Any
import pandas as pd


@dataclass
class LagsProcessor:
    general_config: Dict[str, Any]
    data_config: Dict[str, Any]

    def run(self, data: pd.DataFrame) -> pd.DataFrame:
        if self.data_config.introduce_lags.run:
            lag_introducer = LagIntroducer(
                data=data,
                timestamp=self.data_config.timestamp,
                features=self.data_config.introduce_lags.features,
                lags=self.data_config.introduce_lags.lags,
                optimise_lags=self.data_config.introduce_lags.optimise_lags,
                target=self.data_config.introduce_lags.target,
                max_lag=self.data_config.introduce_lags.max_lag,
                overwrite_existing_features=self.data_config.introduce_lags.overwrite_existing_features,
            )
            data = lag_introducer.run()
            self.generate_artifacts_for_introduce_lags(data)
        return data

    def generate_artifacts_for_introduce_lags(self, data: pd.DataFrame) -> None:
        paths_dict = {
            "time_series_plots": paths.Paths.TIME_SERIES_PLOTS_FOR_LAGGED_FEATURES_PATH.value,
            "histogram_plots": paths.Paths.HISTOGRAM_PLOTS_FOR_LAGGED_FEATURES_PATH.value,
            "custom_plots": paths.Paths.CUSTOM_PLOTS_FOR_LAGGED_FEATURES_PATH.value,
        }
        generate_artifacts(
            self.general_config, self.data_config, data, "stage_6_lags_introduced", paths_dict
        )
