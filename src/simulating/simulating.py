import config.paths as paths
from src.data.generate_simulation import generate_simulations

def override_values_in_clusters(clusters):
    
    # No overrides are currently occuring

    return clusters

def generating_simulations_iron_concentrate_perc_model(best_models, cluster_centers, config):
    simulation_results = generate_simulations(
        features=config.iron_concentrate_perc.model.training_features,
        feature_values_to_simulate=config.simulation.feed_blend_and_controllables_model.feature_values_to_simulate,
        model_choice=config.iron_concentrate_perc.model.model_choice,
        best_models=best_models,
        confidence_interval=config.simulation.feed_blend_and_controllables_model.confidence_interval,
        cluster_centers=cluster_centers,
        feed_blend_and_reagent_modelling=True,
        reagent_features=config.clustering.controllables_model.training_features
        #path=paths.Paths.FEED_BLEND_AND_CONTROLLABLES_IRON_CONCENTRATE_PERC_SIMULATIONS_FILE.value,
        #shapley_path=paths.Paths.SHAPLEY_RESULTS_FILE.value,
        #shapley_plots_path=paths.Paths.SHAPLEY_PLOTS_PATH.value,
    )
    return simulation_results

def generating_simulations_iron_concentrate_perc_feed_blend_model(best_models, cluster_centers, config):
    simulation_results = generate_simulations(
        features=config.iron_concentrate_perc.model.feed_blend_training_features,
        feature_values_to_simulate=config.simulation.feed_blend_model.feature_values_to_simulate,
        model_choice=config.iron_concentrate_perc.model.model_choice,
        best_models=best_models,
        confidence_interval=config.simulation.feed_blend_model.confidence_interval,
        cluster_centers=cluster_centers,
        informational_features=config.clustering.feed_blend_model.informational_features,
        feed_blend_and_reagent_modelling=False
        #path=paths.Paths.FEED_BLEND_IRON_CONCENTRATE_PERC_SIMULATIONS_FILE.value,
        #shapley_path=paths.Paths.SHAPLEY_RESULTS_FILE.value,
        #shapley_plots_path=paths.Paths.SHAPLEY_PLOTS_PATH.value,
    )
    return simulation_results

def generating_simulations_silica_concentrate_perc_model(best_models, cluster_centers, config):
    simulation_results = generate_simulations(
        features=config.silica_concentrate_perc.model.training_features,
        feature_values_to_simulate=config.simulation.feed_blend_and_controllables_model.feature_values_to_simulate,
        model_choice=config.silica_concentrate_perc.model.model_choice,
        best_models=best_models,
        confidence_interval=config.simulation.feed_blend_and_controllables_model.confidence_interval,
        cluster_centers=cluster_centers,
        feed_blend_and_reagent_modelling=True,
        reagent_features=config.clustering.controllables_model.training_features
        #path=paths.Paths.FEED_BLEND_AND_CONTROLLABLES_SILICA_CONCENTRATE_PERC_SIMULATIONS_FILE.value,
        #shapley_path=paths.Paths.SHAPLEY_RESULTS_FILE.value,
        #shapley_plots_path=paths.Paths.SHAPLEY_PLOTS_PATH.value,
    )
    return simulation_results

def generating_simulations_silica_concentrate_perc_feed_blend_model(best_models, cluster_centers, config):
    simulation_results = generate_simulations(
        features=config.silica_concentrate_perc.model.feed_blend_training_features,
        feature_values_to_simulate=config.simulation.feed_blend_model.feature_values_to_simulate,
        model_choice=config.silica_concentrate_perc.model.model_choice,
        best_models=best_models,
        confidence_interval=config.simulation.feed_blend_model.confidence_interval,
        cluster_centers=cluster_centers,
        informational_features=config.clustering.feed_blend_model.informational_features,
        feed_blend_and_reagent_modelling=False
        #path=paths.Paths.FEED_BLEND_SILICA_CONCENTRATE_PERC_SIMULATIONS_FILE.value,
        #shapley_path=paths.Paths.SHAPLEY_RESULTS_FILE.value,
        #shapley_plots_path=paths.Paths.SHAPLEY_PLOTS_PATH.value,
    )
    return simulation_results

def combine_feed_blend_and_controllables_simulations(iron_concentrate_perc_simulation_results,
                                                     silica_concentrate_perc_simulation_results,
                                                     config):

    # Combining the simulations
    iron_concentrate_perc_simulation_results = iron_concentrate_perc_simulation_results.rename(columns={'mean_simulated_predictions': 'IRON_CONCENTRATE_PERC_mean_simulated_predictions'})
    silica_concentrate_perc_simulation_results = silica_concentrate_perc_simulation_results.rename(columns={'mean_simulated_predictions': 'SILICA_CONCENTRATE_PERC_mean_simulated_predictions'})

    iron_concentrate_perc_simulation_results['SILICA_CONCENTRATE_PERC_mean_simulated_predictions'] = silica_concentrate_perc_simulation_results['SILICA_CONCENTRATE_PERC_mean_simulated_predictions']

    # Outputting to a csv file
    iron_concentrate_perc_simulation_results.to_csv(paths.Paths.FEED_BLEND_AND_CONTROLLABLES_SIMULATIONS_FILE.value, index=False)

    return iron_concentrate_perc_simulation_results

def combine_feed_blend_simulations(iron_concentrate_perc_feed_blend_simulation_results,
                                   silica_concentrate_perc_feed_blend_simulation_results,
                                   config):

    # Identiying the historical predictions from each of the feed blend simulations
    iron_concentrate_perc_feed_blend_simulation_results = iron_concentrate_perc_feed_blend_simulation_results.rename(columns={'mean_historical_predictions': 'IRON_CONCENTRATE_PERC_mean_historical_predictions'})
    silica_concentrate_perc_feed_blend_simulation_results = silica_concentrate_perc_feed_blend_simulation_results.rename(columns={'mean_historical_predictions': 'SILICA_CONCENTRATE_PERC_mean_historical_predictions'})

    # Combining the simulations
    iron_concentrate_perc_feed_blend_simulation_results['SILICA_CONCENTRATE_PERC_mean_historical_predictions'] = silica_concentrate_perc_feed_blend_simulation_results['SILICA_CONCENTRATE_PERC_mean_historical_predictions']

    # Outputting to a csv file
    iron_concentrate_perc_feed_blend_simulation_results.to_csv(paths.Paths.FEED_BLEND_SIMULATIONS_FILE.value, index=False)

    return iron_concentrate_perc_feed_blend_simulation_results
