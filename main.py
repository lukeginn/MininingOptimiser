import logging as logger
import warnings
from typing import Any, Dict
from src.setup.classes.setup import Setup
from src.data.reading.classes.data_reader import DataReader
from src.data.preprocessing.classes.missing_data_processor import MissingDataProcessor
from src.data.preprocessing.classes.outlier_processor import OutlierProcessor
from src.data.preprocessing.classes.lags_processor import LagsProcessor
from src.data.preprocessing.classes.data_aggregation_processor import DataAggregationProcessor
from src.data.preprocessing.classes.feature_engineering import FeatureEngineering
from src.data.preprocessing.classes.shutdowns import ShutdownFilter
from src.model.classes.univariable_feature_importance_generator import (
    UnivariableFeatureImportanceGenerator,
)
from src.model.classes.feature_selection import FeatureSelection
from src.model.classes.correlation_matrix_generator import CorrelationMatrixGenerator
from src.model.classes.model_generator import ModelGenerator
from src.model.classes.model_saver import ModelSaver
from src.model.classes.partial_plots_generator import PartialPlotsGenerator
from src.clustering.classes.clustering import Clustering
from src.simulating.classes.simulation_generator import SimulationGenerator
from src.optimising.classes.cluster_optimiser import ClusterOptimiser

logger.basicConfig(level=logger.INFO)
warnings.filterwarnings("ignore")


def main() -> None:
    """Main function to run the data processing and model pipeline."""
    logger.info("Pipeline started")

    setup_instance = Setup()
    general_config = setup_instance.general_config
    data_config = setup_instance.data_config
    model_config = setup_instance.model_config
    clustering_config = setup_instance.clustering_config
    simulation_config = setup_instance.simulation_config
    optimisation_config = setup_instance.optimisation_config

    data = run_data_processing(general_config, data_config)
    model_processors = run_model_training_and_evaluation(data, model_config)
    merged_simulations, merged_feed_blend_simulations, controllables_clusters = (
        run_clustering_and_simulation(
            data, model_processors, clustering_config, simulation_config, model_config
        )
    )
    run_optimisation(
        merged_simulations,
        merged_feed_blend_simulations,
        controllables_clusters,
        clustering_config,
        optimisation_config,
    )

    logger.info("Pipeline completed successfully")


def run_data_processing(
    general_config: Dict[str, Any], data_config: Dict[str, Any]
) -> Any:
    """Run the data processing steps."""
    data_reader = DataReader(general_config, data_config)
    missing_data_processor = MissingDataProcessor(general_config, data_config)
    outlier_processor = OutlierProcessor(general_config, data_config)
    lags_processor = LagsProcessor(general_config, data_config)
    data_aggregator = DataAggregationProcessor(general_config, data_config)
    feature_engineering = FeatureEngineering(general_config, data_config)
    shutdown_filter = ShutdownFilter(general_config, data_config)

    data = data_reader.run()
    data = missing_data_processor.run_identifying_missing_data(data)
    data = outlier_processor.run(data)
    data = missing_data_processor.run_correcting_missing_data(data)
    data = lags_processor.run(data)
    data = data_aggregator.run(data)
    data = feature_engineering.run(data)
    data = missing_data_processor.run_correcting_missing_data_post_aggregation(data)
    data = shutdown_filter.run(data)

    return data


def run_model_training_and_evaluation(
    data: Any, model_config: Dict[str, Any]
) -> Dict[str, Any]:
    """Run model training and evaluation steps."""
    univariable_feature_importance_generator = UnivariableFeatureImportanceGenerator(
        model_config
    )
    feature_selection = FeatureSelection(model_config)
    correlation_matrix_generator = CorrelationMatrixGenerator(model_config)
    model_generator = ModelGenerator(model_config)
    model_saver = ModelSaver(model_config)
    partial_plots_generator = PartialPlotsGenerator(model_config)

    # Initial Model Analytics
    univariable_feature_importance_generator.run_for_iron_concentrate_perc(data)
    univariable_feature_importance_generator.run_for_silica_concentrate_perc(data)

    # Model Training and Evaluation For Iron Concentrate Model (Feed Blend)
    (
        iron_concentrate_perc_feed_blend_training_features,
        iron_concentrate_perc_feed_blend_training_features_per_method,
    ) = feature_selection.run_for_iron_concentrate_perc_feed_blend(data)
    correlation_matrix_generator.run_for_iron_concentrate_perc_feed_blend(
        data, iron_concentrate_perc_feed_blend_training_features_per_method
    )
    iron_concentrate_perc_feed_blend_model = model_generator.run_for_iron_concentrate_perc_feed_blend(
        data, iron_concentrate_perc_feed_blend_training_features
    )
    model_saver.run_for_iron_concentrate_perc_feed_blend(
        iron_concentrate_perc_feed_blend_model
    )
    partial_plots_generator.run_for_iron_concentrate_perc_feed_blend(
        iron_concentrate_perc_feed_blend_model,
        data,
        iron_concentrate_perc_feed_blend_training_features,
    )

    # Model Training and Evaluation For Iron Concentrate Model
    (
        iron_concentrate_perc_training_features,
        iron_concentrate_perc_training_features_per_method,
    ) = feature_selection.run_for_iron_concentrate_perc(data)
    correlation_matrix_generator.run_for_iron_concentrate_perc(
        data, iron_concentrate_perc_training_features_per_method
    )
    iron_concentrate_perc_model = (
        model_generator.run_for_iron_concentrate_perc(
            data, iron_concentrate_perc_training_features
        )
    )
    model_saver.run_for_iron_concentrate_perc(iron_concentrate_perc_model)
    partial_plots_generator.run_for_iron_concentrate_perc(
        iron_concentrate_perc_model, data, iron_concentrate_perc_training_features
    )

    # Model Training and Evaluation For Silica Concentrate Model (Feed Blend)
    (
        silica_concentrate_perc_feed_blend_training_features,
        silica_concentrate_perc_feed_blend_training_features_per_method,
    ) = feature_selection.run_for_silica_concentrate_perc_feed_blend(data)
    correlation_matrix_generator.run_for_silica_concentrate_perc_feed_blend(
        data, silica_concentrate_perc_feed_blend_training_features_per_method
    )
    silica_concentrate_perc_feed_blend_model = model_generator.run_for_silica_concentrate_perc_feed_blend(
        data, silica_concentrate_perc_feed_blend_training_features
    )
    model_saver.run_for_silica_concentrate_perc_feed_blend(
        silica_concentrate_perc_feed_blend_model
    )
    partial_plots_generator.run_for_silica_concentrate_perc_feed_blend(
        silica_concentrate_perc_feed_blend_model,
        data,
        silica_concentrate_perc_feed_blend_training_features,
    )

    # Model Training and Evaluation For Silica Concentrate Model
    (
        silica_concentrate_perc_training_features,
        silica_concentrate_perc_training_features_per_method,
    ) = feature_selection.run_for_silica_concentrate_perc(data)
    correlation_matrix_generator.run_for_silica_concentrate_perc(
        data, silica_concentrate_perc_training_features_per_method
    )
    silica_concentrate_perc_model = (
        model_generator.run_for_silica_concentrate_perc(
            data, silica_concentrate_perc_training_features
        )
    )
    model_saver.run_for_silica_concentrate_perc(silica_concentrate_perc_model)
    partial_plots_generator.run_for_silica_concentrate_perc(
        silica_concentrate_perc_model, data, silica_concentrate_perc_training_features
    )

    return {
        "iron_concentrate_perc_feed_blend_model": iron_concentrate_perc_feed_blend_model,
        "silica_concentrate_perc_feed_blend_model": silica_concentrate_perc_feed_blend_model,
        "iron_concentrate_perc_model": iron_concentrate_perc_model,
        "silica_concentrate_perc_model": silica_concentrate_perc_model,
    }


def run_clustering_and_simulation(
    data: Any,
    models: Dict[str, Any],
    clustering_config: Dict[str, Any],
    simulation_config: Dict[str, Any],
    model_config: Dict[str, Any],
) -> tuple:
    """Run clustering and simulation steps."""
    clustering = Clustering(clustering_config)
    simulation_generator = SimulationGenerator(
        model_config, clustering_config, simulation_config
    )

    feed_blend_clusters = clustering.run_for_feed_blends(data)
    controllables_clusters = clustering.run_for_controllables(data)
    merged_clusters = clustering.run_to_merge_clusters(
        feed_blend_clusters, controllables_clusters
    )

    iron_concentrate_perc_feed_blend_simulations = (
        simulation_generator.run_for_iron_concentrate_perc_feed_blend(
            models["iron_concentrate_perc_feed_blend_model"], feed_blend_clusters
        )
    )
    silica_concentrate_perc_feed_blend_simulations = (
        simulation_generator.run_for_silica_concentrate_perc_feed_blend(
            models["silica_concentrate_perc_feed_blend_model"], feed_blend_clusters
        )
    )
    merged_feed_blend_simulations = (
        simulation_generator.run_to_merge_feed_blend_simulations(
            iron_concentrate_perc_feed_blend_simulations,
            silica_concentrate_perc_feed_blend_simulations,
        )
    )

    iron_concentrate_perc_simulations = (
        simulation_generator.run_for_iron_concentrate_perc(
            models["iron_concentrate_perc_model"], merged_clusters
        )
    )
    silica_concentrate_perc_simulations = (
        simulation_generator.run_for_silica_concentrate_perc(
            models["silica_concentrate_perc_model"], merged_clusters
        )
    )
    merged_simulations = (
        simulation_generator.run_to_merge_feed_blend_and_controllables_simulations(
            iron_concentrate_perc_simulations, silica_concentrate_perc_simulations
        )
    )

    return merged_simulations, merged_feed_blend_simulations, controllables_clusters


def run_optimisation(
    merged_simulations: Any,
    merged_feed_blend_simulations: Any,
    controllables_clusters: Any,
    clustering_config: Dict[str, Any],
    optimisation_config: Dict[str, Any],
) -> None:
    """Run the optimisation step."""
    cluster_optimiser = ClusterOptimiser(clustering_config, optimisation_config)
    optimised_clusters = cluster_optimiser.run(
        merged_simulations, merged_feed_blend_simulations, controllables_clusters
    )


if __name__ == "__main__":
    main()
