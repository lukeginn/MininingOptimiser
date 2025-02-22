import pandas as pd
import logging as logger
from scipy.spatial.distance import cdist
from shared.model.classes.inference_processor import InferenceProcessor


def create_optimal_historical_data(
    optimal_clusters,
    data,
    model,
    model_choice,
    model_target,
    model_training_features,
    feed_blend_features,
    controllable_features,
):

    data = _identify_closest_feed_blend_cluster(
        data, optimal_clusters, feed_blend_features
    )

    data = _apply_optimal_controllable_features(
        data, optimal_clusters, controllable_features
    )

    data = _generate_historical_optimised_target(
        data,
        model,
        model_choice,
        model_target,
        model_training_features,
        feed_blend_features,
        controllable_features,
    )

    return data


def _identify_closest_feed_blend_cluster(data, optimal_clusters, feed_blend_features):
    data_feed_blends = data[feed_blend_features]
    cluster_feed_blends = optimal_clusters[
        [f"{feature}_historical_actuals" for feature in feed_blend_features]
    ]

    distances = cdist(data_feed_blends, cluster_feed_blends, metric="euclidean")
    closest_clusters = distances.argmin(axis=1)

    data["closest_feed_blend_cluster"] = closest_clusters
    if "cluster" in data.columns:
        data.drop("cluster", axis=1, inplace=True)

    return data


def _apply_optimal_controllable_features(data, optimal_clusters, controllable_features):

    for feature in controllable_features:
        data[f"{feature}_simulations"] = data["closest_feed_blend_cluster"].apply(
            lambda x: optimal_clusters.loc[x, f"{feature}_simulations"]
        )

    return data


def _generate_historical_optimised_target(
    data,
    model,
    model_choice,
    model_target,
    model_training_features,
    feed_blend_features,
    controllable_features,
):
    inference_processor = InferenceProcessor(model_choice=model_choice)
    historical_predictions = inference_processor.run(
        models=model,
        data=data[model_training_features],
    )
    optimised_predictions = inference_processor.run(
        models=model,
        data=data[
            feed_blend_features
            + [f"{feature}_simulations" for feature in controllable_features]
        ].rename(columns=lambda x: x.replace("_simulations", "")),
    )

    data[f"{model_target}_historical_predictions"] = historical_predictions
    data[f"{model_target}_optimised_predictions"] = optimised_predictions
    data[f"{model_target}_optimisation_uplift"] = (
        optimised_predictions - historical_predictions
    )
    data[f"{model_target}_with_optimised_uplift_applied"] = (
        data[model_target] + data[f"{model_target}_optimisation_uplift"]
    )

    return data
