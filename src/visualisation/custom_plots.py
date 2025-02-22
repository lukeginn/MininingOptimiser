import numpy as np
import pandas as pd
import logging as logger
import os
import matplotlib.pyplot as plt


def custom_plots(data, path):

    # Custom plots can be placed here

    optimisation_difference_time_series_plots(data, path)

    return data


def optimisation_difference_time_series_plots(data, path):

    if "IRON_CONCENTRATE_PERC_mean_optimisation_difference" in data.columns:
        plt.figure()

        plt.plot(
            data["DATE"],
            data["IRON_CONCENTRATE_PERC_mean_optimised"],
            label="Optimised",
            color="purple",
        )
        plt.plot(
            data["DATE"],
            data["IRON_CONCENTRATE_PERC_mean"],
            label="Historical",
        )

        plt.title(f"Time series plot of Iron Concentrate Percentage Improvement")
        plt.xlabel("Date")
        plt.ylabel("Iron Concentrate Percentage")
        plt.legend()
        plt.savefig(
            os.path.join(path, "iron_concentrate_perc_improvement_time_series.png")
        )
        plt.close()

    if "SILICA_CONCENTRATE_PERC_mean_optimisation_difference" in data.columns:
        plt.figure()

        plt.plot(
            data["DATE"],
            data["SILICA_CONCENTRATE_PERC_mean_optimised"],
            label="Optimised",
            color="purple",
        )
        plt.plot(
            data["DATE"],
            data["SILICA_CONCENTRATE_PERC_mean"],
            label="Historical",
        )

        plt.title(f"Time series plot of Silica Concentrate Percentage Improvement")
        plt.xlabel("Date")
        plt.ylabel("Silica Concentrate Percentage")
        plt.legend()
        plt.savefig(
            os.path.join(path, "silica_concentrate_perc_improvement_time_series.png")
        )
        plt.close()
