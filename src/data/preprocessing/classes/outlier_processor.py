import config.paths as paths
from shared.data.outlier_identifier import OutlierIdentifier
from src.utils.generate_artifacts import generate_artifacts
from dataclasses import dataclass
from typing import Dict, Any
import pandas as pd


@dataclass
class OutlierProcessor:
    general_config: Dict[str, Any]
    data_config: Dict[str, Any]

    def run(self, data: pd.DataFrame) -> pd.DataFrame:
        if self.data_config.identify_outliers.run:
            outlier_identifier = OutlierIdentifier(
                data=data,
                method=self.data_config.identify_outliers.method,
                iqr_threshold=self.data_config.identify_outliers.iqr_threshold,
                z_score_threshold=self.data_config.identify_outliers.z_score_threshold,
                mad_threshold=self.data_config.identify_outliers.mad_threshold,
                dbscan_eps=self.data_config.identify_outliers.dbscan_eps,
                dbscan_min_samples=self.data_config.identify_outliers.dbscan_min_samples,
                isolation_forest_threshold=self.data_config.identify_outliers.isolation_forest_threshold,
                lof_threshold=self.data_config.identify_outliers.lof_threshold,
            )
            data = outlier_identifier.run()
            self.generate_artifacts_for_identifying_outliers(data)
        return data

    def generate_artifacts_for_identifying_outliers(self, data: pd.DataFrame) -> None:
        paths_dict = {
            "time_series_plots": paths.Paths.TIME_SERIES_PLOTS_FOR_OUTLIERS_IDENTIFIED_PATH.value,
            "histogram_plots": paths.Paths.HISTOGRAM_PLOTS_FOR_OUTLIERS_IDENTIFIED_PATH.value,
            "custom_plots": paths.Paths.CUSTOM_PLOTS_FOR_OUTLIERS_IDENTIFIED_PATH.value,
        }
        generate_artifacts(
            self.general_config, data, "stage_3_outliers_identified", paths_dict
        )
