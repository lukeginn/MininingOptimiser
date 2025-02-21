import pandas as pd
import logging as logger
import matplotlib.pyplot as plt


def generate_time_series_plots(data, timestamp, path, features=None):
    if features is None:
        features = numeric_only_features(data=data)

    logger.info("Generating time series plots")
    for feature in features:
        plot_time_series(data=data, feature=feature, timestamp=timestamp, path=path)

    logger.info("Time series plots generated successfully")


def numeric_only_features(data):
    return data.select_dtypes(include="number").columns


def plot_time_series(data, feature, timestamp, path):
    plt.figure()
    data.set_index(timestamp)[feature].plot()
    plt.title(f"Time series plot of {feature}")
    plt.xlabel(timestamp)
    plt.ylabel(feature)
    plt.savefig(f"{path}\{feature}_time_series_plot.png")
    plt.close()
    logger.info(f"Time series plot of {feature} saved successfully")
