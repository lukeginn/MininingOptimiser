import config.paths as paths
from shared.model.generate_clusters import run_clustering
from src.clustering.functions.merging import merging_clusters

class Clustering:
    def __init__(self, config):
        self.config = config

    def run_for_feed_blends(self, data):
        clusters = run_clustering(
            data=data,
            training_features=self.config.clustering.feed_blend_model.training_features,
            informational_features=self.config.clustering.feed_blend_model.informational_features,
            include_row_count_sum=self.config.clustering.feed_blend_model.include_row_count_sum,
            path=paths.Paths.FEED_BLEND_CLUSTERING_FILE.value,
            model_choice=self.config.clustering.feed_blend_model.model_choice,
            k_means_n_clusters=self.config.clustering.feed_blend_model.k_means_n_clusters,
            k_means_max_iter=self.config.clustering.feed_blend_model.k_means_max_iter,
            dbscan_eps=self.config.clustering.feed_blend_model.dbscan_eps,
            dbscan_min_samples=self.config.clustering.feed_blend_model.dbscan_min_samples,
            agglomerative_n_clusters=self.config.clustering.feed_blend_model.agglomerative_n_clusters,
            random_state=self.config.clustering.feed_blend_model.random_state
        )
        return clusters

    def run_for_controllables(self, data):
        clusters = run_clustering(
            data=data,
            training_features=self.config.clustering.controllables_model.training_features,
            informational_features=self.config.clustering.controllables_model.informational_features,
            include_row_count_sum=self.config.clustering.controllables_model.include_row_count_sum,
            path=paths.Paths.CONTROLLABLES_CLUSTERING_FILE.value,
            model_choice=self.config.clustering.controllables_model.model_choice,
            k_means_n_clusters=self.config.clustering.controllables_model.k_means_n_clusters,
            k_means_max_iter=self.config.clustering.controllables_model.k_means_max_iter,
            dbscan_eps=self.config.clustering.controllables_model.dbscan_eps,
            dbscan_min_samples=self.config.clustering.controllables_model.dbscan_min_samples,
            agglomerative_n_clusters=self.config.clustering.controllables_model.agglomerative_n_clusters,
            random_state=self.config.clustering.controllables_model.random_state
        )
        return clusters

    def run_to_merge_clusters(self, feed_blend_clusters, controllables_clusters):
        merged_clusters = merging_clusters(
            feed_blend_clusters=feed_blend_clusters,
            controllables_clusters=controllables_clusters,
            feed_blend_training_features=self.config.clustering.feed_blend_model.training_features,
            controllables_training_features=self.config.clustering.controllables_model.training_features,
            path=paths.Paths.COMBINED_CLUSTERING_FILE.value
        )
        return merged_clusters