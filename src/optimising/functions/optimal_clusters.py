import pandas as pd
import logging as logger


def create_optimised_clusters(
    cluster_combination_centers,
    feed_blend_simulations,
    controllables_clusters,
    path,
    feature_to_optimize,
    optimisation_direction,
    controllable_features,
    constraint_features,
    constraint_limits_per_feature,
):

    feature_to_optimize += "_simulated_predictions"

    # Engineer New Feature
    # Engineer features that may be useful as informtion variables or constraints or as the feature to optimise

    # Apply constraints
    constrained_clusters = cluster_combination_centers.copy()
    constraint_features = [
        feature + "_simulated_predictions" for feature in constraint_features
    ]
    for feature, limits in zip(constraint_features, constraint_limits_per_feature):
        constrained_clusters = constrained_clusters[
            (constrained_clusters[feature] >= limits[0])
            & (constrained_clusters[feature] <= limits[1])
        ]

    # Optimise Clusters
    if optimisation_direction:
        optimal_clusters = constrained_clusters.loc[
            constrained_clusters.groupby("feed_blend_cluster_id")[
                feature_to_optimize
            ].idxmax()
        ]
    elif optimisation_direction:
        optimal_clusters = constrained_clusters.loc[
            constrained_clusters.groupby("feed_blend_cluster_id")[
                feature_to_optimize
            ].idxmin()
        ]

    # Merge controllable_features using 'controllables_cluster_id' and 'cluster' as the joining keys
    controllable_features = [
        feature + "_historical_actuals" for feature in controllable_features
    ]
    controllable_features.append("cluster")

    optimal_clusters = optimal_clusters.merge(
        feed_blend_simulations[controllable_features],
        left_on="feed_blend_cluster_id",
        right_on="cluster",
        how="left",
    ).drop(columns=["cluster"])

    # Bring all of the features with the suffix '_historical_actuals' to the beginning of the dataframe
    historical_actuals_features = [
        col for col in optimal_clusters.columns if col.endswith("_historical_actuals")
    ]
    optimal_clusters = optimal_clusters[
        ["feed_blend_cluster_id", "controllables_cluster_id"]
        + historical_actuals_features
        + [
            col
            for col in optimal_clusters.columns
            if col not in historical_actuals_features
        ]
    ]

    # Remove the duplicate 'feed_blend_cluster_id' and 'controllables_cluster_id' columns
    optimal_clusters = optimal_clusters.loc[:, ~optimal_clusters.columns.duplicated()]

    # Merge with target features from feed_blend_clusters
    features_to_merge = [
        "IRON_CONCENTRATE_PERC_mean_historical_predictions",
        "SILICA_CONCENTRATE_PERC_mean_historical_predictions",
        "IRON_CONCENTRATE_PERC_mean_historical_actuals",
        "SILICA_CONCENTRATE_PERC_mean_historical_actuals",
    ]
    optimal_clusters = pd.concat(
        [
            optimal_clusters.reset_index(drop=True),
            feed_blend_simulations[features_to_merge].reset_index(drop=True),
        ],
        axis=1,
    )

    if path:
        optimal_clusters.to_csv(path)
        logger.info(f"Optimal clusters exported to {path}")

    return optimal_clusters
