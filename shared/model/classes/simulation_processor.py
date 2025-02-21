import pandas as pd
import logging as logger
import matplotlib.pyplot as plt
from shared.model.classes.inference_processor import InferenceProcessor
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Tuple


@dataclass
class SimulationProcessor:
    model: List[Any]
    model_choice: str
    features: List[str]
    confidence_interval: bool
    cluster_centers: Optional[pd.DataFrame] = None
    feature_values_to_simulate: Optional[Dict[str, List[float]]] = None
    path: Optional[str] = None
    informational_features: Optional[List[str]] = None
    feed_blend_and_controllables_modelling: bool = False
    controllables_features: Optional[List[str]] = None

    def run(self) -> pd.DataFrame:
        logger.info("Generating simulations")

        simulation_data = self._generating_simulation_data()

        simulation_results = (
            self._generating_simulation_predictions_for_all_simulation_data(
                simulation_data=simulation_data
            )
        )

        simulation_results = self._merge_simulation_results_with_cluster_centers(
            simulation_results=simulation_results
        )

        if self.informational_features is not None:
            informational_features = [
                feature + "_historical_actuals"
                for feature in self.informational_features
            ]
            informational_features = [
                feature
                for feature in informational_features
                if feature not in simulation_results.columns
            ]

            simulation_results = pd.concat(
                [simulation_results, self.cluster_centers[informational_features]],
                axis=1,
            )
            # Reorder the dataset so that 'mean_historical_actuals' is the last column
            columns = [
                col
                for col in simulation_results.columns
                if col != "mean_historical_predictions"
            ]
            columns.append("mean_historical_predictions")
            simulation_results = simulation_results[columns]

        if self.feed_blend_and_controllables_modelling:
            simulation_results.columns = [
                col.replace("_historical_predictions", "_simulated_predictions")
                for col in simulation_results.columns
            ]

            controllables_features = [
                feature + "_historical_actuals"
                for feature in self.controllables_features
            ]
            simulation_results.rename(
                columns={
                    feature: feature.replace("_historical_actuals", "_simulations")
                    for feature in controllables_features
                },
                inplace=True,
            )

        if self.path is not None:
            self._export_simulation_results(
                simulation_results=simulation_results,
                path=self.path,
            )
        logger.info("Simulations generated successfully")

        return simulation_results

    def _generating_simulation_data(self) -> pd.DataFrame:
        features = [feature + "_historical_actuals" for feature in self.features]

        if self.cluster_centers is not None:
            cluster_centers_to_simulate = self._process_cluster_centers(
                self.cluster_centers, features
            )

        if self.cluster_centers is not None:
            simulated_data = self._generate_simulation_data(features)

        if (
            self.cluster_centers is not None
            and self.feature_values_to_simulate is not None
        ):
            # Combine the cluster centers and the simulation data
            simulation_data = pd.concat(
                [cluster_centers_to_simulate, simulated_data]
            ).reset_index(drop=True)
        elif self.cluster_centers is not None:
            simulation_data = cluster_centers_to_simulate
        elif self.feature_values_to_simulate is not None:
            simulation_data = simulated_data
        else:
            raise ValueError(
                "Either cluster_centers or feature_values_to_simulate must be provided"
            )

        simulation_data.columns = [
            col.replace("_historical_actuals", "") for col in simulation_data.columns
        ]

        return simulation_data

    def _generate_simulation_data(self, features: List[str]) -> pd.DataFrame:
        simulation_data = pd.DataFrame(
            self.feature_values_to_simulate, columns=features
        )
        logger.info("Simulation data generated successfully")

        return simulation_data

    def _process_cluster_centers(
        self, cluster_centers: pd.DataFrame, features: List[str]
    ) -> pd.DataFrame:
        logger.info("Processing cluster centers")

        # Remove duplicate columns
        cluster_centers = cluster_centers.loc[:, ~cluster_centers.columns.duplicated()]

        cluster_id_features = [
            feature for feature in cluster_centers.columns if "cluster_id" in feature
        ]
        features = cluster_id_features + [
            feature for feature in features if feature not in cluster_id_features
        ]
        cluster_centers = pd.DataFrame(cluster_centers, columns=features)
        return cluster_centers

    def _generating_simulation_predictions_for_all_simulation_data(
        self, simulation_data: pd.DataFrame
    ) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
        results = []
        for index, one_simulation in simulation_data.iterrows():
            one_simulation = one_simulation.to_frame().T

            simulation_predictions = self._generating_simulation_predictions(
                one_simulation=one_simulation
            )

            # Merge the input and the predictions
            one_simulation = pd.concat(
                [
                    one_simulation.reset_index(drop=True),
                    simulation_predictions.reset_index(drop=True),
                ],
                axis=1,
            )
            results.append(one_simulation)

        simulation_results = pd.concat(results).reset_index(drop=True)

        return simulation_results

    def _generating_simulation_predictions(
        self, one_simulation: pd.DataFrame
    ) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
        inference_processor = InferenceProcessor(model_choice=self.model_choice)
        simulation_predictions = inference_processor.run(
            models=self.model, data=one_simulation[self.features]
        )
        simulation_predictions = pd.DataFrame(simulation_predictions).T
        if self.confidence_interval:
            simulation_predictions.columns = ["lower", "mean", "upper"]
        else:
            simulation_predictions.columns = ["mean"]

        return simulation_predictions

    def _merge_simulation_results_with_cluster_centers(
        self, simulation_results: pd.DataFrame
    ) -> pd.DataFrame:

        cluster_centers = self.cluster_centers.reset_index().rename(
            columns={"index": "cluster_number"}
        )
        if all(
            col in cluster_centers.columns
            for col in ["cluster", "row_count", "row_count_proportion"]
        ):
            cluster_centers = cluster_centers[
                ["cluster", "row_count", "row_count_proportion"]
            ]
            simulation_results = pd.concat(
                [cluster_centers, simulation_results], axis=1
            )

        columns_to_modify = [
            col
            for col in simulation_results.columns
            if col
            not in [
                "cluster",
                "row_count",
                "row_count_proportion",
                "mean",
                "feed_blend_cluster_id",
                "controllables_cluster_id",
            ]
        ]
        simulation_results.rename(
            columns={col: col + "_historical_actuals" for col in columns_to_modify},
            inplace=True,
        )
        simulation_results.rename(
            columns={"mean": "mean_historical_predictions"}, inplace=True
        )

        return simulation_results

    def _export_simulation_results(
        self, simulation_results: pd.DataFrame, path: str
    ) -> None:
        simulation_results.to_csv(path, index=False)
        logger.info(f"Simulation results exported to {path}")
