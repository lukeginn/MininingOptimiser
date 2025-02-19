import config.paths as paths
from shared.model.generate_clusters import run_clustering
from src.clustering.merging import merging_clusters

def generating_clusters_feed_blend_model(data, config):
    clusters = run_clustering(
        data=data, 
        training_features=config.clustering.feed_blend_model.training_features,
        informational_features=config.clustering.feed_blend_model.informational_features,
        include_row_count_sum=config.clustering.feed_blend_model.include_row_count_sum,
        path=paths.Paths.FEED_BLEND_CLUSTERING_FILE.value,
        model_choice=config.clustering.feed_blend_model.model_choice,
        k_means_n_clusters=config.clustering.feed_blend_model.k_means_n_clusters,
        k_means_max_iter=config.clustering.feed_blend_model.k_means_max_iter,
        dbscan_eps=config.clustering.feed_blend_model.dbscan_eps,
        dbscan_min_samples=config.clustering.feed_blend_model.dbscan_min_samples,
        agglomerative_n_clusters=config.clustering.feed_blend_model.agglomerative_n_clusters,
        random_state=config.clustering.feed_blend_model.random_state
    )
    return clusters

def generating_clusters_controllables_model(data, config):
    clusters = run_clustering(
        data=data,
        training_features=config.clustering.controllables_model.training_features,  
        informational_features=config.clustering.controllables_model.informational_features,
        include_row_count_sum=config.clustering.controllables_model.include_row_count_sum,
        path=paths.Paths.CONTROLLABLES_CLUSTERING_FILE.value,
        model_choice=config.clustering.controllables_model.model_choice,
        k_means_n_clusters=config.clustering.controllables_model.k_means_n_clusters,
        k_means_max_iter=config.clustering.controllables_model.k_means_max_iter,
        dbscan_eps=config.clustering.controllables_model.dbscan_eps,
        dbscan_min_samples=config.clustering.controllables_model.dbscan_min_samples,
        agglomerative_n_clusters=config.clustering.controllables_model.agglomerative_n_clusters,
        random_state=config.clustering.controllables_model.random_state
    )
    return clusters

def merging_feed_blend_and_controllables_clusters(feed_blend_clusters, reagent_clusters, config):
    cluster_combination_centers = merging_clusters(
        feed_blend_clusters = feed_blend_clusters,
        reagent_clusters = reagent_clusters,
        feed_blend_training_features = config.clustering.feed_blend_model.training_features,
        reagent_training_features = config.clustering.controllables_model.training_features,
        path = paths.Paths.COMBINED_CLUSTERING_FILE.value
    )

    return cluster_combination_centers