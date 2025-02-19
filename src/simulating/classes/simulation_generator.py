import config.paths as paths
from src.simulating.functions.generate_simulation import generate_simulations
from dataclasses import dataclass
from typing import List, Dict, Any
import pandas as pd


@dataclass
class SimulationGenerator:
    model_config: Dict[str, Any]
    clustering_config: Dict[str, Any]
    simulation_config: Dict[str, Any]

    def run_for_iron_concentrate_perc(
        self, best_models: List[Any], cluster_centers: pd.DataFrame
    ) -> pd.DataFrame:
        # If needed, we override the values in the clusters to ensure that the simulations are tailored to answer the business questions
        cluster_centers = self.override_values_in_clusters(cluster_centers)

        simulation_results = generate_simulations(
            features=self.model_config.iron_concentrate_perc.model.training_features,
            feature_values_to_simulate=self.simulation_config.feed_blend_and_controllables_model.feature_values_to_simulate,
            model_choice=self.model_config.iron_concentrate_perc.model.model_choice,
            best_models=best_models,
            confidence_interval=self.simulation_config.feed_blend_and_controllables_model.confidence_interval,
            cluster_centers=cluster_centers,
            feed_blend_and_controllables_modelling=True,
            controllables_features=self.clustering_config.controllables_model.training_features,
        )
        return simulation_results

    def run_for_iron_concentrate_perc_feed_blend(
        self, best_models: List[Any], cluster_centers: pd.DataFrame
    ) -> pd.DataFrame:
        simulation_results = generate_simulations(
            features=self.model_config.iron_concentrate_perc.model.feed_blend_training_features,
            feature_values_to_simulate=self.simulation_config.feed_blend_model.feature_values_to_simulate,
            model_choice=self.model_config.iron_concentrate_perc.model.model_choice,
            best_models=best_models,
            confidence_interval=self.simulation_config.feed_blend_model.confidence_interval,
            cluster_centers=cluster_centers,
            informational_features=self.clustering_config.feed_blend_model.informational_features,
            feed_blend_and_controllables_modelling=False,
        )
        return simulation_results

    def run_for_silica_concentrate_perc(
        self, best_models: List[Any], cluster_centers: pd.DataFrame
    ) -> pd.DataFrame:
        # If needed, we override the values in the clusters to ensure that the simulations are tailored to answer the business questions
        cluster_centers = self.override_values_in_clusters(cluster_centers)

        simulation_results = generate_simulations(
            features=self.model_config.silica_concentrate_perc.model.training_features,
            feature_values_to_simulate=self.simulation_config.feed_blend_and_controllables_model.feature_values_to_simulate,
            model_choice=self.model_config.silica_concentrate_perc.model.model_choice,
            best_models=best_models,
            confidence_interval=self.simulation_config.feed_blend_and_controllables_model.confidence_interval,
            cluster_centers=cluster_centers,
            feed_blend_and_controllables_modelling=True,
            controllables_features=self.clustering_config.controllables_model.training_features,
        )
        return simulation_results

    def run_for_silica_concentrate_perc_feed_blend(
        self, best_models: List[Any], cluster_centers: pd.DataFrame
    ) -> pd.DataFrame:
        simulation_results = generate_simulations(
            features=self.model_config.silica_concentrate_perc.model.feed_blend_training_features,
            feature_values_to_simulate=self.simulation_config.feed_blend_model.feature_values_to_simulate,
            model_choice=self.model_config.silica_concentrate_perc.model.model_choice,
            best_models=best_models,
            confidence_interval=self.simulation_config.feed_blend_model.confidence_interval,
            cluster_centers=cluster_centers,
            informational_features=self.clustering_config.feed_blend_model.informational_features,
            feed_blend_and_controllables_modelling=False,
        )
        return simulation_results

    def run_to_merge_feed_blend_and_controllables_simulations(
        self,
        iron_concentrate_perc_simulation_results: pd.DataFrame,
        silica_concentrate_perc_simulation_results: pd.DataFrame,
    ) -> pd.DataFrame:
        # Merging the simulations
        iron_concentrate_perc_simulation_results = iron_concentrate_perc_simulation_results.rename(
            columns={
                "mean_simulated_predictions": "IRON_CONCENTRATE_PERC_mean_simulated_predictions"
            }
        )
        silica_concentrate_perc_simulation_results = silica_concentrate_perc_simulation_results.rename(
            columns={
                "mean_simulated_predictions": "SILICA_CONCENTRATE_PERC_mean_simulated_predictions"
            }
        )

        iron_concentrate_perc_simulation_results[
            "SILICA_CONCENTRATE_PERC_mean_simulated_predictions"
        ] = silica_concentrate_perc_simulation_results[
            "SILICA_CONCENTRATE_PERC_mean_simulated_predictions"
        ]

        # Outputting to a csv file
        iron_concentrate_perc_simulation_results.to_csv(
            paths.Paths.FEED_BLEND_AND_CONTROLLABLES_SIMULATIONS_FILE.value, index=False
        )

        return iron_concentrate_perc_simulation_results

    def run_to_merge_feed_blend_simulations(
        self,
        iron_concentrate_perc_feed_blend_simulation_results: pd.DataFrame,
        silica_concentrate_perc_feed_blend_simulation_results: pd.DataFrame,
    ) -> pd.DataFrame:
        # Identifying the historical predictions from each of the feed blend simulations
        iron_concentrate_perc_feed_blend_simulation_results = iron_concentrate_perc_feed_blend_simulation_results.rename(
            columns={
                "mean_historical_predictions": "IRON_CONCENTRATE_PERC_mean_historical_predictions"
            }
        )
        silica_concentrate_perc_feed_blend_simulation_results = silica_concentrate_perc_feed_blend_simulation_results.rename(
            columns={
                "mean_historical_predictions": "SILICA_CONCENTRATE_PERC_mean_historical_predictions"
            }
        )

        # Merging the simulations
        iron_concentrate_perc_feed_blend_simulation_results[
            "SILICA_CONCENTRATE_PERC_mean_historical_predictions"
        ] = silica_concentrate_perc_feed_blend_simulation_results[
            "SILICA_CONCENTRATE_PERC_mean_historical_predictions"
        ]

        # Outputting to a csv file
        iron_concentrate_perc_feed_blend_simulation_results.to_csv(
            paths.Paths.FEED_BLEND_SIMULATIONS_FILE.value, index=False
        )

        return iron_concentrate_perc_feed_blend_simulation_results

    def override_values_in_clusters(self, clusters: pd.DataFrame) -> pd.DataFrame:
        # No overrides are currently occurring
        return clusters
