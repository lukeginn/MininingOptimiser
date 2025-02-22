import config.paths as paths
from src.optimising.functions.optimal_historical_data import (
    create_optimal_historical_data,
)
from src.utils.generate_artifacts import generate_artifacts
from dataclasses import dataclass
from typing import Dict, Any
import pandas as pd


@dataclass
class HistoricalOptimiser:
    data: pd.DataFrame
    models: Dict[str, Any]
    general_config: Dict[str, Any]
    data_config: Dict[str, Any]
    model_config: Dict[str, Any]
    clustering_config: Dict[str, Any]

    def run(self, optimal_clusters: pd.DataFrame) -> pd.DataFrame:
        self.data = create_optimal_historical_data(
            optimal_clusters=optimal_clusters,
            data=self.data,
            model=self.models["iron_concentrate_perc_model"],
            model_choice=self.model_config.iron_concentrate_perc.model.model_choice,
            model_target=self.model_config.iron_concentrate_perc.model.target,
            model_training_features=self.model_config.iron_concentrate_perc.model.training_features,
            feed_blend_features=self.clustering_config.feed_blend_model.training_features,
            controllable_features=self.clustering_config.controllables_model.training_features,
        )
        self.data = create_optimal_historical_data(
            optimal_clusters=optimal_clusters,
            data=self.data,
            model=self.models["silica_concentrate_perc_model"],
            model_choice=self.model_config.silica_concentrate_perc.model.model_choice,
            model_target=self.model_config.silica_concentrate_perc.model.target,
            model_training_features=self.model_config.silica_concentrate_perc.model.training_features,
            feed_blend_features=self.clustering_config.feed_blend_model.training_features,
            controllable_features=self.clustering_config.controllables_model.training_features,
        )
        return self.data

    def generate_artifacts_for_optimised_data(self) -> None:
        paths_dict = {
            "time_series_plots": paths.Paths.TIME_SERIES_PLOTS_FOR_OPTIMISED_DATA_PATH.value,
            "histogram_plots": paths.Paths.HISTOGRAM_PLOTS_FOR_OPTIMISED_DATA_PATH.value,
            "custom_plots": paths.Paths.CUSTOM_PLOTS_FOR_OPTIMISED_DATA_PATH.value,
        }
        generate_artifacts(
            self.general_config,
            self.data_config,
            self.data,
            "stage_11_optimised_data",
            paths_dict,
        )
