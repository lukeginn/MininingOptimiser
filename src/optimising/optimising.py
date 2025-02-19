import config.paths as paths
from src.model.optimal_clusters import create_optimised_clusters

def optimise_clusters(cluster_combination_centers, feed_blend_simulations, controllables_clusters, config):

    optimal_clusters = create_optimised_clusters(
        cluster_combination_centers=cluster_combination_centers,
        feed_blend_simulations=feed_blend_simulations,
        controllables_clusters=controllables_clusters,
        path=paths.Paths.OPTIMISED_CLUSTERS_FILE.value,
        feature_to_optimize=config.optimisation.feature_to_optimise,
        optimisation_direction=config.optimisation.direction_to_optimise,
        controllable_features=config.clustering.controllables_model.training_features,
        constraint_features=config.optimisation.constraints.features,
        constraint_limits_per_feature=config.optimisation.constraints.limits
    )

    return optimal_clusters