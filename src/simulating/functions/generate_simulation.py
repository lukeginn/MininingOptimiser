import pandas as pd
import logging as logger
from shared.model.generate_model import generate_predictions
import matplotlib.pyplot as plt


def generate_simulations(
    features,
    model_choice,
    best_models,
    confidence_interval,
    cluster_centers=None,
    feature_values_to_simulate=None,
    path=None,
    shapley_path=None,
    shapley_plots_path=None,
    informational_features=None,
    feed_blend_and_controllables_modelling=False,
    controllables_features = None
):
    logger.info("Generating simulations")

    simulation_data = generating_simulation_data(
        features=features,
        feature_values_to_simulate=feature_values_to_simulate,
        cluster_centers=cluster_centers,
    )

    simulation_results, shapley_results = generating_simulation_predictions_for_all_simulation_data(
        simulation_data=simulation_data,
        model_choice=model_choice,
        best_models=best_models,
        training_features=features,
        confidence_interval=confidence_interval,
        shapley_values=True if shapley_path is not None else False
    )

    simulation_results = merge_simulation_results_with_cluster_centers(
        simulation_results=simulation_results,
        cluster_centers=cluster_centers,
    )

    if shapley_path is not None:
        shapley_results = merge_simulation_results_with_cluster_centers(
            simulation_results=shapley_results,
            cluster_centers=cluster_centers,
        )
    if shapley_path is not None:
        plot_individual_shapley_values(
            shapley_results=shapley_results,
            features=features,
            path=shapley_plots_path,
        )
    if informational_features is not None:
        informational_features = [feature + '_historical_actuals' for feature in informational_features]
        informational_features = [feature for feature in informational_features if feature not in simulation_results.columns]

        simulation_results = pd.concat([simulation_results, cluster_centers[informational_features]], axis=1)
        # Reorder the dataset so that 'mean_historical_actuals' is the last column
        columns = [col for col in simulation_results.columns if col != 'mean_historical_predictions']
        columns.append('mean_historical_predictions')
        simulation_results = simulation_results[columns]

    if feed_blend_and_controllables_modelling:
        simulation_results.columns = [col.replace('_historical_predictions', '_simulated_predictions') for col in simulation_results.columns]

        controllables_features = [feature + '_historical_actuals' for feature in controllables_features]
        simulation_results.rename(columns={feature: feature.replace('_historical_actuals', '_simulations') for feature in controllables_features}, inplace=True)

    if path is not None:
        export_simulation_results(
            simulation_results=simulation_results,
            path=path,
        )
    if shapley_path is not None:
        export_simulation_results(
            simulation_results=shapley_results,
            path=shapley_path,
        )
    logger.info("Simulations generated successfully")

    return simulation_results

def generating_simulation_data(features, feature_values_to_simulate, cluster_centers):

    features = [feature + '_historical_actuals' for feature in features]

    if cluster_centers is not None:
        cluster_centers_to_simulate = process_cluster_centers(cluster_centers, features)

    if cluster_centers is not None:
        simulated_data = generate_simulation_data(
            features=features, feature_values_to_simulate=feature_values_to_simulate
        )

    if cluster_centers is not None and feature_values_to_simulate is not None:
        # Combine the cluster centers and the simulation data
        simulation_data = pd.concat([cluster_centers_to_simulate, simulated_data]).reset_index(drop=True)
    elif cluster_centers is not None:
        simulation_data = cluster_centers_to_simulate
    elif feature_values_to_simulate is not None:
        simulation_data = simulated_data
    else:
        raise ValueError("Either cluster_centers or feature_values_to_simulate must be provided")
    
    simulation_data.columns = [col.replace('_historical_actuals', '') for col in simulation_data.columns]

    return simulation_data

def generate_simulation_data(features, feature_values_to_simulate):
    simulation_data = pd.DataFrame(feature_values_to_simulate, columns=features)
    logger.info("Simulation data generated successfully")

    return simulation_data

def process_cluster_centers(cluster_centers, features):
    logger.info("Processing cluster centers")

    # Remove duplicate columns
    cluster_centers = cluster_centers.loc[:, ~cluster_centers.columns.duplicated()]

    if cluster_centers.shape[1] > 14:
        print()
    cluster_id_features = [feature for feature in cluster_centers.columns if 'cluster_id' in feature]
    features = cluster_id_features + [feature for feature in features if feature not in cluster_id_features]
    cluster_centers = pd.DataFrame(cluster_centers, columns=features)
    return cluster_centers

def generating_simulation_predictions_for_all_simulation_data(simulation_data, model_choice, best_models, training_features, confidence_interval, shapley_values):
    results = []
    shapley_results = []
    for index, one_simulation in simulation_data.iterrows():
        one_simulation = one_simulation.to_frame().T

        simulation_predictions, simulation_shapley_values = generating_simulation_predictions(
            one_simulation=one_simulation,
            model_choice=model_choice,
            best_models=best_models,
            training_features=training_features,
            confidence_interval=confidence_interval,
            shapley_values=shapley_values
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
        if shapley_values:
            shapley_results.append(simulation_shapley_values)

    simulation_results = pd.concat(results).reset_index(drop=True)
    if shapley_values:
        shapley_results = pd.concat(shapley_results).reset_index(drop=True)

    return simulation_results, shapley_results

def generating_simulation_predictions(
    one_simulation, model_choice, best_models, training_features, confidence_interval, shapley_values
):
    
    simulation_predictions = generate_predictions(
        model_choice=model_choice,
        models=best_models,
        data=one_simulation[training_features],
        return_confidence_interval=confidence_interval,
    )

    if shapley_values:
        simulation_shapley_values = generate_predictions(
            model_choice=model_choice,
            models=best_models,
            data=one_simulation[training_features],
            return_shapley_values=True
        )
    else:
        simulation_shapley_values = None

    simulation_predictions = pd.DataFrame(simulation_predictions).T
    if confidence_interval:
        simulation_predictions.columns = ["lower", "mean", "upper"]
    else:
        simulation_predictions.columns = ["mean"]

    return simulation_predictions, simulation_shapley_values

def merge_simulation_results_with_cluster_centers(simulation_results, cluster_centers):

    cluster_centers = cluster_centers.reset_index().rename(columns={'index': 'cluster_number'})
    if all(col in cluster_centers.columns for col in ['cluster', 'row_count', 'row_count_proportion']):
        cluster_centers = cluster_centers[['cluster', 'row_count', 'row_count_proportion']]
        simulation_results = pd.concat([cluster_centers, simulation_results], axis=1)

    columns_to_modify = [col for col in simulation_results.columns if col not in ['cluster', 'row_count', 'row_count_proportion', 'mean', 'feed_blend_cluster_id', 'controllables_cluster_id']]
    simulation_results.rename(columns={col: col + '_historical_actuals' for col in columns_to_modify}, inplace=True)
    simulation_results.rename(columns={'mean': 'mean_historical_predictions'}, inplace=True)

    return simulation_results

def export_simulation_results(simulation_results, path):
    simulation_results.to_csv(path, index=False)
    logger.info(f"Simulation results exported to {path}")

def plot_individual_shapley_values(shapley_results, features, path):
    for index, row in shapley_results.iterrows():
        shap_values = row[features + ['intercept', 'final_prediction']].values
        plt.figure()
        shap.summary_plot(shap_values.reshape(1, -1), features + ['intercept', 'final_prediction'], plot_type="bar", show=False)
        plt.savefig(f'{path}/shapley_values_cluster_{index}.png')
        plt.close()