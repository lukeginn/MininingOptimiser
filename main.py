import logging as logger
import warnings
from src.setup.setup import setup
from src.data.reading.data_reader import DataReader
from src.data.reading.data_reader import DataReader
from src.data.preprocessing.missing_data import identifying_missing_data
from src.data.preprocessing.outliers import identifying_outliers
from src.data.preprocessing.missing_data import correcting_missing_data
from src.data.preprocessing.lags import introducing_lags
from src.data.preprocessing.aggregation import aggregating_data
from src.data.preprocessing.feature_engineering import run_feature_engineering
from src.data.preprocessing.missing_data import correcting_missing_data_post_aggregation
from src.data.preprocessing.shutdowns import run_filter_shutdowns
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
    data = data_reader.read_file()
    
    data = identifying_missing_data(data, config)
    data = identifying_outliers(data, config)
    data = correcting_missing_data(data, config)
    data = introducing_lags(data, config)
    data = aggregating_data(data, config)
    data = run_feature_engineering(data, config)
    data = correcting_missing_data_post_aggregation(data, config)
    data = run_filter_shutdowns(data, config)

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
