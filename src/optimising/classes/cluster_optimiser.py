import config.paths as paths
from src.optimising.functions.optimal_clusters import create_optimised_clusters
from dataclasses import dataclass


@dataclass
class ClusterOptimiser:
    clustering_config: dict
    optimisation_config: dict

    def run(
        self,
        cluster_combination_centers,
        feed_blend_simulations,
        controllables_clusters,
    ):
        optimal_clusters = create_optimised_clusters(
            cluster_combination_centers=cluster_combination_centers,
            feed_blend_simulations=feed_blend_simulations,
            controllables_clusters=controllables_clusters,
            path=paths.Paths.OPTIMISED_CLUSTERS_FILE.value,
            feature_to_optimize=self.optimisation_config.feature_to_optimise,
            optimisation_direction=self.optimisation_config.direction_to_optimise,
            controllable_features=self.clustering_config.controllables_model.training_features,
            constraint_features=self.optimisation_config.constraints.features,
            constraint_limits_per_feature=self.optimisation_config.constraints.limits,
        )
        return optimal_clusters
