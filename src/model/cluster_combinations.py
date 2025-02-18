import numpy as np
import pandas as pd
import logging as logger
import matplotlib.pyplot as plt
import itertools


def create_cluster_combinations(feed_blend_clusters, reagent_clusters, feed_blend_training_features, reagent_training_features, path = None):

    feed_blend_clusters_ids = feed_blend_clusters.index
    reagent_clusters_ids = reagent_clusters.index

    feed_blend_training_features = [feature + '_historical_actuals' for feature in feed_blend_training_features]
    reagent_training_features = [feature + '_historical_actuals' for feature in reagent_training_features]

    # Identify the columns which will not be in the combined_clusters, which are in the feed_blend_clusters
    feed_blend_clusters_columns_not_in_combined = [col for col in feed_blend_clusters.columns if col not in feed_blend_training_features]
    feed_blend_clusters_columns_not_in_combined = [col for col in feed_blend_clusters_columns_not_in_combined if col not in ['row_count', 'row_count_proportion']]

    feed_blend_clusters_filtered = feed_blend_clusters[feed_blend_training_features + feed_blend_clusters_columns_not_in_combined]
    reagent_clusters_filtered = reagent_clusters[
        [feature for feature in reagent_training_features
         if feature not in feed_blend_training_features]
    ]

    combinations = list(itertools.product(feed_blend_clusters_filtered.iterrows(), reagent_clusters_filtered.iterrows()))
    combined_clusters = pd.DataFrame([
        {'feed_blend_cluster_id': feed_blend[0], 'reagent_cluster_id': reagent[0], **feed_blend[1], **reagent[1]}
        for feed_blend, reagent in combinations
    ])

    # Reorder columns
    combined_clusters = combined_clusters[['feed_blend_cluster_id', 'reagent_cluster_id'] + feed_blend_training_features + reagent_training_features + feed_blend_clusters_columns_not_in_combined]

    if path:
        export_combined_cluster_centers(
            combined_cluster_centers=combined_clusters,
            path=path
        )

    return combined_clusters

def export_combined_cluster_centers(combined_cluster_centers, path):
    combined_cluster_centers.to_csv(path)
    logger.info(f"Combined Cluster centers exported to {path}")