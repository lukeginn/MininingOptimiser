import config.paths as paths
from src.utils.generate_artifacts import generate_artifacts
from dataclasses import dataclass
from typing import Dict, Any, Optional
import pandas as pd


@dataclass
class ShutdownFilter:
    general_config: Dict[str, Any]
    data_config: Dict[str, Any]

    def run(self, data: pd.DataFrame) -> pd.DataFrame:
        if self.data_config.filter_shutdowns.run:
            data = self.filter_shutdowns(data)
            self.generate_artifacts_for_run(data)
        return data

    def generate_artifacts_for_run(self, data: pd.DataFrame) -> None:
        paths_dict = {
            "time_series_plots": paths.Paths.TIME_SERIES_PLOTS_FOR_FILTERING_SHUTDOWN_PATH.value,
            "histogram_plots": paths.Paths.HISTOGRAM_PLOTS_FOR_FILTERING_SHUTDOWN_PATH.value,
            "custom_plots": paths.Paths.CUSTOM_PLOTS_FOR_FILTERING_SHUTDOWN_PATH.value,
        }
        generate_artifacts(
            self.general_config,
            self.data_config,
            data,
            "stage_10_filter_shutdowns",
            paths_dict,
        )

    def filter_shutdowns(self, data: pd.DataFrame) -> pd.DataFrame:
        shutdown_dates = self.identify_shutdown_times(data)
        if shutdown_dates is not None:
            data = data[~data["DATE"].isin(shutdown_dates)]
        return data

    def identify_shutdown_times(self, data: pd.DataFrame) -> Optional[pd.Series]:
        # Currently no shutdown periods have been identified
        shutdown_dates = None
        return shutdown_dates
