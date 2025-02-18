import numpy as np
import pandas as pd
import logging as logger
import matplotlib.pyplot as plt


def generate_histogram_plots(data, path, features=None, number_of_bins=30):
    if features is None:
        features = numeric_only_features(data=data)

    if number_of_bins is None:
        number_of_bins_per_feature = determine_optimal_number_of_bins_per_feature(
            data=data, features=features
        )

    logger.info("Generating histogram plots")
    for feature in features:
        if number_of_bins is not None:
            plot_histogram(data=data, feature=feature, path=path, bins=number_of_bins)
        else:
            plot_histogram(
                data=data,
                feature=feature,
                path=path,
                bins=number_of_bins_per_feature[feature],
            )

    logger.info("Histogram plots generated successfully")


def numeric_only_features(data):
    return data.select_dtypes(include="number").columns


def determine_optimal_number_of_bins_per_feature(data, features):
    number_of_bins = {}
    for feature in features:
        number_of_bins[feature] = determine_optimal_number_of_bins(
            data=data, feature=feature
        )
    return number_of_bins


def determine_optimal_number_of_bins(data, feature):
    # Sturges' formula
    number_of_bins = int(1 + 3.322 * np.log10(data[feature].count()))
    return number_of_bins


def plot_histogram(data, feature, path, bins):
    plt.figure()
    data[feature].plot(
        kind="hist", bins=bins, edgecolor="black", color="lightgrey", rwidth=0.9
    )
    plt.title(f"Histogram of {feature}")
    plt.xlabel(feature)
    plt.ylabel("Frequency")
    plt.savefig(f"{path}\{feature}_histogram_plot.png")
    plt.close()
    logger.info(f"Histogram of {feature} saved successfully")
