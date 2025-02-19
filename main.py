import logging as logger
import warnings
from src.setup.setup import setup
from src.data.reading.classes.data_reader import DataReader
from src.data.preprocessing.classes.missing_data_processor import MissingDataProcessor
from src.data.preprocessing.classes.outlier_processor import OutlierProcessor
from src.data.preprocessing.classes.lags_processor import LagsProcessor
from src.data.preprocessing.classes.aggregator import DataAggregator
from src.data.preprocessing.classes.feature_engineering import FeatureEngineering
from src.data.preprocessing.classes.shutdowns import ShutdownFilter
from src.model.classes.univariable_feature_importance_generator import UnivariableFeatureImportanceGenerator
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


def main():
    logger.info("Pipeline started")
    config = setup()

    data_reader = DataReader(config)
    missing_data_processor = MissingDataProcessor(config)
    outlier_processor = OutlierProcessor(config)
    lags_processor = LagsProcessor(config)
    data_aggregator = DataAggregator(config)
    feature_engineering = FeatureEngineering(config)
    shutdown_filter = ShutdownFilter(config)
    univariable_feature_importance_generator = UnivariableFeatureImportanceGenerator(config)
    feature_selection = FeatureSelection(config)
    correlation_matrix_generator = CorrelationMatrixGenerator(config)
    model_generator = ModelGenerator(config)  
    model_saver = ModelSaver(config)
    partial_plots_generator = PartialPlotsGenerator(config)
    clustering = Clustering(config)
    simulation_generator = SimulationGenerator(config)
    cluster_optimiser = ClusterOptimiser(config)

    data = data_reader.run()
    data = missing_data_processor.run_identifying_missing_data(data)
    data = outlier_processor.run(data)
    data = missing_data_processor.run_correcting_missing_data(data)
    data = lags_processor.run(data)
    data = data_aggregator.run(data)
    data = feature_engineering.run(data)
    data = missing_data_processor.run_correcting_missing_data_post_aggregation(data)
    data = shutdown_filter.run(data)

    # Initial Model Analytics
    univariable_feature_importance_generator.run_for_iron_concentrate_perc(data)
    univariable_feature_importance_generator.run_for_silica_concentrate_perc(data)

    # Model Training and Evaluation For Iron Concentrate Model (Feed Blend)
    iron_concentrate_perc_feed_blend_training_features, iron_concentrate_perc_feed_blend_training_features_per_method = feature_selection.run_for_iron_concentrate_perc_feed_blend(data)

    correlation_matrix_generator.run_for_iron_concentrate_perc_feed_blend(data, iron_concentrate_perc_feed_blend_training_features_per_method)
    iron_concentrate_perc_feed_blend_model, best_params, best_rmse, feature_importance = model_generator.run_for_iron_concentrate_perc_feed_blend(
        data, iron_concentrate_perc_feed_blend_training_features
    )
    model_saver.run_for_iron_concentrate_perc_feed_blend(iron_concentrate_perc_feed_blend_model)
    partial_plots_generator.run_for_iron_concentrate_perc_feed_blend(iron_concentrate_perc_feed_blend_model, data, iron_concentrate_perc_feed_blend_training_features)


    # Model Training and Evaluation For Iron Concentrate Model
    iron_concentrate_perc_training_features, iron_concentrate_perc_training_features_per_method = feature_selection.run_for_iron_concentrate_perc(data)
    correlation_matrix_generator.run_for_iron_concentrate_perc(data, iron_concentrate_perc_training_features_per_method)
    iron_concentrate_perc_model, best_params, best_rmse, feature_importance = model_generator.run_for_iron_concentrate_perc(
        data, iron_concentrate_perc_training_features
    )
    model_saver.run_for_iron_concentrate_perc(iron_concentrate_perc_model)
    partial_plots_generator.run_for_iron_concentrate_perc(iron_concentrate_perc_model, data, iron_concentrate_perc_training_features)


    # Model Training and Evaluation For Silica Concentrate Model (Feed Blend)
    silica_concentrate_perc_feed_blend_training_features, silica_concentrate_perc_feed_blend_training_features_per_method = feature_selection.run_for_silica_concentrate_perc_feed_blend(
        data
    )
    correlation_matrix_generator.run_for_silica_concentrate_perc_feed_blend(data, silica_concentrate_perc_feed_blend_training_features_per_method)
    silica_concentrate_perc_feed_blend_model, best_params, best_rmse, feature_importance = model_generator.run_for_silica_concentrate_perc_feed_blend(
        data, silica_concentrate_perc_feed_blend_training_features
    )
    model_saver.run_for_silica_concentrate_perc_feed_blend(silica_concentrate_perc_feed_blend_model)
    partial_plots_generator.run_for_silica_concentrate_perc_feed_blend(silica_concentrate_perc_feed_blend_model, data, silica_concentrate_perc_feed_blend_training_features)


    # Model Training and Evaluation For Silica Concentrate Model
    silica_concentrate_perc_training_features, silica_concentrate_perc_training_features_per_method = feature_selection.run_for_silica_concentrate_perc(
        data
    )
    correlation_matrix_generator.run_for_silica_concentrate_perc(data, silica_concentrate_perc_training_features_per_method)
    silica_concentrate_perc_model, best_params, best_rmse, feature_importance = model_generator.run_for_silica_concentrate_perc(
        data, silica_concentrate_perc_training_features
    )
    model_saver.run_for_silica_concentrate_perc(silica_concentrate_perc_model)
    partial_plots_generator.run_for_silica_concentrate_perc(silica_concentrate_perc_model, data, silica_concentrate_perc_training_features)

    # Clustering
    feed_blend_clusters = clustering.run_for_feed_blends(data)
    controllables_clusters = clustering.run_for_controllables(data)
    merged_clusters = clustering.run_to_merge_clusters(feed_blend_clusters, controllables_clusters)

    # Simulating Feed Blends Only (To Understand The Impact Of Feed Blends)
    iron_concentrate_perc_feed_blend_simulations = simulation_generator.run_for_iron_concentrate_perc_feed_blend(iron_concentrate_perc_feed_blend_model, feed_blend_clusters)
    silica_concentrate_perc_feed_blend_simulations = simulation_generator.run_for_silica_concentrate_perc_feed_blend(silica_concentrate_perc_feed_blend_model, feed_blend_clusters)
    merged_feed_blend_simulations = simulation_generator.run_to_merge_feed_blend_simulations(
        iron_concentrate_perc_feed_blend_simulations,
        silica_concentrate_perc_feed_blend_simulations
    )

    # Simulating Feed Blends And Controllables (To Identify Optimal Controllables Per Feed Blend)
    iron_concentrate_perc_simulations = simulation_generator.run_for_iron_concentrate_perc(iron_concentrate_perc_model, merged_clusters)
    silica_concentrate_perc_simulations = simulation_generator.run_for_silica_concentrate_perc(silica_concentrate_perc_model, merged_clusters)
    merged_simulations = simulation_generator.run_to_merge_feed_blend_and_controllables_simulations(
        iron_concentrate_perc_simulations,
        silica_concentrate_perc_simulations
    )

    # Optimising The Simulation Results
    optimised_clusters = cluster_optimiser.run(merged_simulations, merged_feed_blend_simulations, controllables_clusters)

    logger.info("Pipeline completed successfully")


if __name__ == "__main__":
    main()
