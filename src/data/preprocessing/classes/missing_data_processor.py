import config.paths as paths
from shared.data.classes.missing_data_identifier import MissingDataIdentifier
from shared.data.classes.missing_data_corrector import MissingDataCorrector
from src.utils.generate_artifacts import generate_artifacts
from dataclasses import dataclass
from typing import Dict, Any
import pandas as pd


@dataclass
class MissingDataProcessor:
    general_config: Dict[str, Any]
    data_config: Dict[str, Any]

    def run_identifying_missing_data(self, data: pd.DataFrame) -> pd.DataFrame:
        if self.data_config.identify_missing_data.run:
            missing_data_identifier = MissingDataIdentifier(
                data=data,
                timestamp=self.data_config.timestamp,
                features=self.data_config.identify_missing_data.features,
                unique_values_identification=self.data_config.identify_missing_data.unique_values.run,
                unique_values_threshold=self.data_config.identify_missing_data.unique_values.threshold,
                explicit_missing_values=self.data_config.identify_missing_data.explicit_missing_values.run,
                explicit_missing_indicators=self.data_config.identify_missing_data.explicit_missing_values.indicators,
                repeating_values=self.data_config.identify_missing_data.repeating_values.run,
                repeating_values_threshold=self.data_config.identify_missing_data.repeating_values.threshold,
                repeating_values_proportion_threshold=self.data_config.identify_missing_data.repeating_values.proportion_threshold,
            )
            data = missing_data_identifier.run()
            self.generate_artifacts_for_identifying_missing_data(data)
        return data

    def run_correcting_missing_data(self, data: pd.DataFrame) -> pd.DataFrame:
        if self.data_config.correct_missing_data.run:
            missing_data_corrector = MissingDataCorrector(
                data=data,
                delete_all_rows_with_missing_values=self.data_config.correct_missing_data.delete_all_rows_with_missing_values.run,
                interpolate_time_series=self.data_config.correct_missing_data.interpolate_time_series.run,
                interpolate_highly_regular_time_series=self.data_config.correct_missing_data.interpolate_highly_regular_time_series.run,
                replace_missing_values_with_x=self.data_config.correct_missing_data.replace_missing_values_with_x.run,
                replace_missing_values_with_last_known_value=self.data_config.correct_missing_data.replace_missing_values_with_last_known_value.run,
                interpolate_method=self.data_config.correct_missing_data.interpolate_time_series.method,
                interpolate_limit_direction=self.data_config.correct_missing_data.interpolate_time_series.limit_direction,
                interpolate_max_gap=self.data_config.correct_missing_data.interpolate_time_series.max_gap,
                interpolate_highly_regular_method=self.data_config.correct_missing_data.interpolate_highly_regular_time_series.method,
                interpolate_highly_regular_limit_direction=self.data_config.correct_missing_data.interpolate_highly_regular_time_series.limit_direction,
                interpolate_highly_regular_interval_min=self.data_config.correct_missing_data.interpolate_highly_regular_time_series.regular_interval_min,
                replace_missing_values_with_last_known_value_backfill=self.data_config.correct_missing_data.replace_missing_values_with_last_known_value.backfill,
                timestamp=self.data_config.timestamp,
                x=self.data_config.correct_missing_data.replace_missing_values_with_x.x,
            )
            data = missing_data_corrector.run()
            self.generate_artifacts_for_correcting_missing_data(data)
        return data

    def run_correcting_missing_data_post_aggregation(
        self, data: pd.DataFrame
    ) -> pd.DataFrame:
        if self.data_config.correct_missing_data_after_aggregation.run:
            missing_data_corrector = MissingDataCorrector(
                data=data,
                delete_all_rows_with_missing_values=self.data_config.correct_missing_data_after_aggregation.delete_all_rows_with_missing_values.run,
                interpolate_time_series=self.data_config.correct_missing_data_after_aggregation.interpolate_time_series.run,
                interpolate_highly_regular_time_series=self.data_config.correct_missing_data_after_aggregation.interpolate_highly_regular_time_series.run,
                replace_missing_values_with_x=self.data_config.correct_missing_data_after_aggregation.replace_missing_values_with_x.run,
                replace_missing_values_with_last_known_value=self.data_config.correct_missing_data_after_aggregation.replace_missing_values_with_last_known_value.run,
                interpolate_method=self.data_config.correct_missing_data_after_aggregation.interpolate_time_series.method,
                interpolate_limit_direction=self.data_config.correct_missing_data_after_aggregation.interpolate_time_series.limit_direction,
                interpolate_max_gap=self.data_config.correct_missing_data_after_aggregation.interpolate_time_series.max_gap,
                interpolate_highly_regular_method=self.data_config.correct_missing_data_after_aggregation.interpolate_highly_regular_time_series.method,
                interpolate_highly_regular_limit_direction=self.data_config.correct_missing_data_after_aggregation.interpolate_highly_regular_time_series.limit_direction,
                interpolate_highly_regular_interval_min=self.data_config.correct_missing_data_after_aggregation.interpolate_highly_regular_time_series.regular_interval_min,
                replace_missing_values_with_last_known_value_backfill=self.data_config.correct_missing_data_after_aggregation.replace_missing_values_with_last_known_value.backfill,
                timestamp=self.data_config.timestamp,
                x=self.data_config.correct_missing_data_after_aggregation.replace_missing_values_with_x.x,
            )
            data = missing_data_corrector.run()
            self.generate_artifacts_for_correcting_missing_data_post_aggregation(data)
        return data

    def generate_artifacts_for_identifying_missing_data(
        self, data: pd.DataFrame
    ) -> None:
        paths_dict = {
            "time_series_plots": paths.Paths.TIME_SERIES_PLOTS_FOR_AGGREGATED_FEATURES_PATH.value,
            "histogram_plots": paths.Paths.HISTOGRAM_PLOTS_FOR_AGGREGATED_FEATURES_PATH.value,
            "custom_plots": paths.Paths.CUSTOM_PLOTS_FOR_AGGREGATED_FEATURES_PATH.value,
        }
        generate_artifacts(
            self.general_config,
            self.data_config,
            data,
            "stage_2_missing_data_identified",
            paths_dict,
        )

    def generate_artifacts_for_correcting_missing_data(
        self, data: pd.DataFrame
    ) -> None:
        paths_dict = {
            "time_series_plots": paths.Paths.TIME_SERIES_PLOTS_FOR_MISSING_DATA_CORRECTED_PATH.value,
            "histogram_plots": paths.Paths.HISTOGRAM_PLOTS_FOR_MISSING_DATA_CORRECTED_PATH.value,
            "custom_plots": paths.Paths.CUSTOM_PLOTS_FOR_MISSING_DATA_CORRECTED_PATH.value,
        }
        generate_artifacts(
            self.general_config,
            self.data_config,
            data,
            "stage_5_missing_data_corrected",
            paths_dict,
        )

    def generate_artifacts_for_correcting_missing_data_post_aggregation(
        self, data: pd.DataFrame
    ) -> None:
        paths_dict = {
            "time_series_plots": paths.Paths.TIME_SERIES_PLOTS_FOR_MISSING_DATA_CORRECTED_POST_AGGREGATION_PATH.value,
            "histogram_plots": paths.Paths.HISTOGRAM_PLOTS_FOR_MISSING_DATA_CORRECTED_POST_AGGREGATION_PATH.value,
            "custom_plots": paths.Paths.CUSTOM_PLOTS_FOR_MISSING_DATA_CORRECTED_POST_AGGREGATION_PATH.value,
        }
        generate_artifacts(
            self.general_config,
            self.data_config,
            data,
            "stage_9_missing_data_corrected_post_aggregation",
            paths_dict,
        )
