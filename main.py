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
from src.model.univariabe_feature_importance import generating_univariabe_feature_importance_for_iron_concentrate_perc_model
from src.model.univariabe_feature_importance import generating_univariabe_feature_importance_for_silica_concentrate_perc_model
from src.model.feature_selection import run_feature_selection_iron_concentrate_perc_feed_blend_model
from src.model.feature_selection import run_feature_selection_iron_concentrate_perc_model
from src.model.feature_selection import run_feature_selection_silica_concentrate_perc_feed_blend_model
from src.model.feature_selection import run_feature_selection_silica_concentrate_perc_model
from src.model.correlation_matrix import generating_correlation_matrix_iron_concentrate_perc_feed_blend_model
from src.model.correlation_matrix import generating_correlation_matrix_iron_concentrate_perc_model
from src.model.correlation_matrix import generating_correlation_matrix_silica_concentrate_perc_feed_blend_model
from src.model.correlation_matrix import generating_correlation_matrix_silica_concentrate_perc_model
from src.model.train_model import generating_model_iron_concentrate_perc_feed_blend_model
from src.model.train_model import generating_model_iron_concentrate_perc_model
from src.model.train_model import generating_model_silica_concentrate_perc_feed_blend_model
from src.model.train_model import generating_model_silica_concentrate_perc_model
from src.model.save_model import saving_models_iron_concentrate_perc_feed_blend_model
from src.model.save_model import saving_models_iron_concentrate_perc_model
from src.model.save_model import saving_models_silica_concentrate_perc_feed_blend_model
from src.model.save_model import saving_models_silica_concentrate_perc_model
from src.model.partial_plots import generating_partial_plots_iron_concentrate_perc_feed_blend_model
from src.model.partial_plots import generating_partial_plots_iron_concentrate_perc_model
from src.model.partial_plots import generating_partial_plots_silica_concentrate_perc_feed_blend_model
from src.model.partial_plots import generating_partial_plots_silica_concentrate_perc_model
from src.clustering.clustering import generating_clusters_feed_blend_model
from src.clustering.clustering import generating_clusters_controllables_model
from src.clustering.clustering import merging_feed_blend_and_controllables_clusters
from src.simulating.simulating import override_values_in_clusters
from src.simulating.simulating import generating_simulations_iron_concentrate_perc_feed_blend_model
from src.simulating.simulating import generating_simulations_silica_concentrate_perc_feed_blend_model
from src.simulating.simulating import combine_feed_blend_simulations
from src.simulating.simulating import generating_simulations_iron_concentrate_perc_model
from src.simulating.simulating import generating_simulations_silica_concentrate_perc_model
from src.simulating.simulating import combine_feed_blend_and_controllables_simulations
from src.optimising.optimising import optimise_clusters

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

    data = data_reader.read_file()
    data = missing_data_processor.identifying_missing_data(data)
    data = outlier_processor.identifying_outliers(data)
    data = missing_data_processor.correcting_missing_data(data)
    data = lags_processor.introduce_lags(data)
    data = data_aggregator.aggregate_data(data)
    data = feature_engineering.run(data)
    data = missing_data_processor.correcting_missing_data_post_aggregation(data)
    data = shutdown_filter.run(data)

    # Initial Model Analytics
    generating_univariabe_feature_importance_for_iron_concentrate_perc_model(data, config)
    generating_univariabe_feature_importance_for_silica_concentrate_perc_model(data, config)

    # Model Training and Evaluation For Iron Concentrate Model (Feed Blend)
    iron_concentrate_perc_feed_blend_model_training_features, iron_concentrate_perc_feed_blend_model_training_features_per_method = run_feature_selection_iron_concentrate_perc_feed_blend_model(
        data, config
    )
    generating_correlation_matrix_iron_concentrate_perc_feed_blend_model(data, iron_concentrate_perc_feed_blend_model_training_features_per_method, config)
    iron_concentrate_perc_feed_blend_model, best_params, best_rmse, feature_importance = generating_model_iron_concentrate_perc_feed_blend_model(
        data, iron_concentrate_perc_feed_blend_model_training_features, config
    )
    saving_models_iron_concentrate_perc_feed_blend_model(iron_concentrate_perc_feed_blend_model, config)
    generating_partial_plots_iron_concentrate_perc_feed_blend_model(iron_concentrate_perc_feed_blend_model, data, iron_concentrate_perc_feed_blend_model_training_features, config)


    # Model Training and Evaluation For Iron Concentrate Model
    iron_concentrate_perc_model_training_features, iron_concentrate_perc_model_training_features_per_method = run_feature_selection_iron_concentrate_perc_model(
        data, config
    )
    generating_correlation_matrix_iron_concentrate_perc_model(data, iron_concentrate_perc_model_training_features_per_method, config)
    iron_concentrate_perc_model, best_params, best_rmse, feature_importance = generating_model_iron_concentrate_perc_model(
        data, iron_concentrate_perc_model_training_features, config
    )
    saving_models_iron_concentrate_perc_model(iron_concentrate_perc_model, config)
    generating_partial_plots_iron_concentrate_perc_model(iron_concentrate_perc_model, data, iron_concentrate_perc_model_training_features, config)


    # Model Training and Evaluation For Silica Concentrate Model (Feed Blend)
    silica_concentrate_perc_feed_blend_model_training_features, silica_concentrate_perc_feed_blend_model_training_features_per_method = run_feature_selection_silica_concentrate_perc_feed_blend_model(
        data, config
    )
    generating_correlation_matrix_silica_concentrate_perc_feed_blend_model(data, silica_concentrate_perc_feed_blend_model_training_features_per_method, config)
    silica_concentrate_perc_feed_blend_model, best_params, best_rmse, feature_importance = generating_model_silica_concentrate_perc_feed_blend_model(
        data, silica_concentrate_perc_feed_blend_model_training_features, config
    )
    saving_models_silica_concentrate_perc_feed_blend_model(silica_concentrate_perc_feed_blend_model, config)
    generating_partial_plots_silica_concentrate_perc_feed_blend_model(silica_concentrate_perc_feed_blend_model, data, silica_concentrate_perc_feed_blend_model_training_features, config)


    # Model Training and Evaluation For Silica Concentrate Model
    silica_concentrate_perc_model_training_features, silica_concentrate_perc_model_training_features_per_method = run_feature_selection_silica_concentrate_perc_model(
        data, config
    )
    generating_correlation_matrix_silica_concentrate_perc_model(data, silica_concentrate_perc_model_training_features_per_method, config)
    silica_concentrate_perc_model, best_params, best_rmse, feature_importance = generating_model_silica_concentrate_perc_model(
        data, silica_concentrate_perc_model_training_features, config
    )
    saving_models_silica_concentrate_perc_model(silica_concentrate_perc_model, config)
    generating_partial_plots_silica_concentrate_perc_model(silica_concentrate_perc_model, data, silica_concentrate_perc_model_training_features, config)

    # Clustering
    feed_blend_clusters = generating_clusters_feed_blend_model(data, config)
    controllables_clusters = generating_clusters_controllables_model(data, config)
    combined_clusters = merging_feed_blend_and_controllables_clusters(feed_blend_clusters, controllables_clusters, config)
    combined_clusters = override_values_in_clusters(combined_clusters)

    # Simulating Feed Blends Only (To Understand The Impact Of Feed Blends)
    iron_concentrate_perc_feed_blend_simulation_results = generating_simulations_iron_concentrate_perc_feed_blend_model(iron_concentrate_perc_feed_blend_model, feed_blend_clusters, config)
    silica_concentrate_perc_feed_blend_simulation_results = generating_simulations_silica_concentrate_perc_feed_blend_model(silica_concentrate_perc_feed_blend_model, feed_blend_clusters, config)
    combined_feed_blend_simulation_results = combine_feed_blend_simulations(
        iron_concentrate_perc_feed_blend_simulation_results,
        silica_concentrate_perc_feed_blend_simulation_results,
        config
    )

    # Simulating Feed Blends And Controllables (To Identify Optimal Controllables Per Feed Blend)
    iron_concentrate_perc_simulation_results = generating_simulations_iron_concentrate_perc_model(iron_concentrate_perc_model, combined_clusters, config)
    silica_concentrate_perc_simulation_results = generating_simulations_silica_concentrate_perc_model(silica_concentrate_perc_model, combined_clusters, config)
    combined_simulation_results = combine_feed_blend_and_controllables_simulations(
        iron_concentrate_perc_simulation_results,
        silica_concentrate_perc_simulation_results,
        config
    )

    # Optimising The Simulation Results
    optimised_clusters = optimise_clusters(combined_simulation_results, combined_feed_blend_simulation_results, controllables_clusters, config)

    logger.info("Pipeline completed successfully")


if __name__ == "__main__":
    main()
