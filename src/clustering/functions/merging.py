import pandas as pd
import logging as logger
import itertools


def merging_clusters(
    feed_blend_clusters,
    controllables_clusters,
    feed_blend_training_features,
    controllables_training_features,
    path=None,
):

    feed_blend_training_features = [
        feature + "_historical_actuals" for feature in feed_blend_training_features
    ]
    controllables_training_features = [
        feature + "_historical_actuals" for feature in controllables_training_features
    ]

    # Identify the columns which will not be in the merged_clusters, which are in the feed_blend_clusters
    feed_blend_clusters_columns_not_in_combined = [
        col
        for col in feed_blend_clusters.columns
        if col not in feed_blend_training_features
    ]
    feed_blend_clusters_columns_not_in_combined = [
        col
        for col in feed_blend_clusters_columns_not_in_combined
        if col not in ["row_count", "row_count_proportion"]
    ]

    feed_blend_clusters_filtered = feed_blend_clusters[
        feed_blend_training_features + feed_blend_clusters_columns_not_in_combined
    ]
    controllables_clusters_filtered = controllables_clusters[
        [
            feature
            for feature in controllables_training_features
            if feature not in feed_blend_training_features
        ]
    ]

    combinations = list(
        itertools.product(
            feed_blend_clusters_filtered.iterrows(),
            controllables_clusters_filtered.iterrows(),
        )
    )
    merged_clusters = pd.DataFrame(
        [
            {
                "feed_blend_cluster_id": feed_blend[0],
                "controllables_cluster_id": controllables[0],
                **feed_blend[1],
                **controllables[1],
            }
            for feed_blend, controllables in combinations
        ]
    )

    # Reorder columns
    merged_clusters = merged_clusters[
        ["feed_blend_cluster_id", "controllables_cluster_id"]
        + feed_blend_training_features
        + controllables_training_features
        + feed_blend_clusters_columns_not_in_combined
    ]

    if path:
        export_merged_clusters(merged_clusters=merged_clusters, path=path)

    return merged_clusters


def export_merged_clusters(merged_clusters, path):
    merged_clusters.to_csv(path)
    logger.info(f"Merged clusters exported to {path}")
