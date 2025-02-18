import numpy as np
import pandas as pd
import logging as logger
import matplotlib.pyplot as plt


def custom_plots_for_clusters(data, clusters, training_features, informational_features, path):

    data_aggregated = aggregate_data_for_each_cluster(data=data)
    custom_cluster_scatter_plot(
        x=data_aggregated['surface_tph_proportion'],
        y=data_aggregated['e48_tph_proportion'] + data_aggregated['e26_tph_proportion'],
        x_label='surface_tph_proportion',
        y_label='underground_tph_proportion',
        cluster_labels=data_aggregated.index,
        path=path
    )
    
    features_to_box_and_whisker_plot = ['cu_recovery_perc_pims_rec_cu_b_mean'] + training_features
    for feature in features_to_box_and_whisker_plot:
        custom_cluster_box_and_whisker(
            data=data,
            cluster_label='cluster',
            y_feature=feature,
            y2_feature_values=data_aggregated['e48_tph_proportion'] + data_aggregated['e26_tph_proportion'],
            path=path
        )

    return data

def find_closest_cluster_for_each_row(data, clusters, training_features):
    
    clusters_filtered = clusters[training_features]
    data_filtered = data[training_features]
    data['closest_cluster'] = data_filtered.apply(
        lambda row: np.argmin(np.linalg.norm(clusters_filtered.values - row.values, axis=1)),
        axis=1
    )

    return data

def aggregate_data_for_each_cluster(data):
    
    data_aggregated = data.groupby('cluster').mean()

    return data_aggregated

def custom_cluster_scatter_plot(x,y,x_label,y_label,cluster_labels, path):

    plt.figure(figsize=(10, 6))
    plt.scatter(x, y, c='black')
    for i, label in enumerate(cluster_labels):
        plt.text(x[i] + 0.005, y[i] + 0.005, str(label), fontsize=8, color='black', ha='right')
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(f'{x_label} vs {y_label} by Cluster')
    plt.savefig(path / f'{x_label}_vs_{y_label}_by_cluster.png')
    plt.close()

def custom_cluster_box_and_whisker(data,cluster_label,y_feature,y2_feature_values,path):

    boxprops = dict(linestyle='-', linewidth=1.5, color='black')
    medianprops = dict(linestyle='-', linewidth=1.5, color='black')
    plt.figure(figsize=(12, 8))
    sorted_clusters = sorted(data[cluster_label].unique())
    data_to_plot = [data[data[cluster_label] == cluster][y_feature] for cluster in sorted_clusters]
    plt.boxplot(data_to_plot, boxprops=boxprops, medianprops=medianprops, showfliers=False)
    plt.xlabel('Clusters')
    plt.ylabel(y_feature)
    plt.title(y_feature)
    plt.xticks(ticks=np.arange(1, len(sorted_clusters) + 1), labels=sorted_clusters)
    ax1 = plt.gca()
    ax2 = ax1.twinx()
    y2_scaled = (y2_feature_values - y2_feature_values.min()) / (y2_feature_values.max() - y2_feature_values.min()) * (data[y_feature].max() - data[y_feature].min()) + data[y_feature].min()
    
    for i, cluster in enumerate(sorted_clusters):
        y2_value = y2_scaled[i]
        ax2.plot(i + 1, y2_value, 'o', color='orange')
    
    ax2.set_ylabel('underground_tph_proportion')
    #ax2.set_yticklabels(y2_feature_values)
    ax2.set_yticks(np.linspace(y2_scaled.min(), y2_scaled.max(), num=5))
    ax2.set_yticklabels([f'{val:.2f}' for val in np.linspace(y2_feature_values.min(), y2_feature_values.max(), num=5)])
    ax2.tick_params(axis='y')
    plt.savefig(path / f'box_and_whisker_{y_feature}_each_cluster.png')
    plt.close()