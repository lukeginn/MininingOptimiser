import config.paths as paths
from src.model.optimal_clusters import create_optimised_clusters

class ClusterOptimiser:
    def __init__(self, config):
        self.config = config

    def run(self, cluster_combination_centers, feed_blend_simulations, controllables_clusters):
        optimal_clusters = create_optimised_clusters(
            cluster_combination_centers=cluster_combination_centers,
            feed_blend_simulations=feed_blend_simulations,
            controllables_clusters=controllables_clusters,
            path=paths.Paths.OPTIMISED_CLUSTERS_FILE.value,
            feature_to_optimize=self.config.optimisation.feature_to_optimise,
            optimisation_direction=self.config.optimisation.direction_to_optimise,
            controllable_features=self.config.clustering.controllables_model.training_features,
            constraint_features=self.config.optimisation.constraints.features,
            constraint_limits_per_feature=self.config.optimisation.constraints.limits
        )
        return optimal_clusters