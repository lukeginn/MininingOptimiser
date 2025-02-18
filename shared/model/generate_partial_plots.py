import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt
import logging as logger
import os
from sklearn.inspection import partial_dependence


def generate_partial_plots(
    model_choice,
    models,
    features,
    data,
    path,
    plot_confidence_interval=True,
    plot_feature_density=True,
    degree_of_smoothing=3,
    plot_feature_density_as_histogram=True,
    number_of_bins_in_histogram=10,
    grid_resolution=100
):

    partial_dependence_df_per_feature = generating_partial_dependence_data(
        model_choice=model_choice,
        models=models,
        data=data,
        features=features,
        generate_feature_density=plot_feature_density,
        degree_of_smoothing=degree_of_smoothing,
        grid_resolution=grid_resolution
    )

    writing_partial_dependence_data(
        partial_dependence_df_per_feature=partial_dependence_df_per_feature, path=path
    )

    generating_partial_plots(
        partial_dependence_df_per_feature=partial_dependence_df_per_feature,
        path=path,
        data=data,
        plot_confidence_interval=plot_confidence_interval,
        plot_feature_density=plot_feature_density,
        plot_feature_density_as_histogram=plot_feature_density_as_histogram,
        number_of_bins_in_histogram=number_of_bins_in_histogram,
    )

    return None


def generating_partial_dependence_data(
    model_choice, models, data, features, generate_feature_density, degree_of_smoothing, grid_resolution
):

    logger.info(f"Generating partial dependence data for model choice: {model_choice}")
    if model_choice == "linear_regression":
        partial_dependence_df_per_feature = (
            generate_partial_dependence_for_linear_regression(
                models=models, data=data, features=features, grid_resolution=grid_resolution
            )
        )
    elif model_choice == "gam":
        partial_dependence_df_per_feature = generate_partial_dependence_for_gam(
            models=models, features=features, grid_resolution=grid_resolution
        )
    elif model_choice == "gbm":
        partial_dependence_df_per_feature = generate_partial_dependence_for_gbm(
            models=models, features=features, data=data, grid_resolution=grid_resolution
        )
    else:
        print("Model choice not supported for partial plots")
        return None

    if generate_feature_density:
        logger.info("Generating feature density")
        partial_dependence_df_per_feature = generating_feature_density(
            partial_dependence_df_per_feature=partial_dependence_df_per_feature,
            data=data,
            degree_of_smoothing=degree_of_smoothing,
        )

    logger.info("Partial dependence data generated successfully")
    return partial_dependence_df_per_feature


def generating_feature_density(
    partial_dependence_df_per_feature, data, degree_of_smoothing
):

    # Add the density of the feature in the raw data to the partial dependence data per feature
    for i, partial_dependence_df in enumerate(partial_dependence_df_per_feature):
        feature = partial_dependence_df.columns[0]
        feature_values = data[feature]
        density = np.histogram(
            feature_values, bins=len(partial_dependence_df), density=True
        )
        density_y = gaussian_filter1d(density[0], sigma=degree_of_smoothing)
        density_y = (
            density_y
            / density_y.max()
            * partial_dependence_df["partial_dependence"].max()
        )
        partial_dependence_df["feature_density"] = density_y

    return partial_dependence_df_per_feature


def generate_partial_dependence_for_linear_regression(models, data, features, grid_resolution):
    partial_dependence_df_per_feature = []

    for feature in features:
        feature_values = np.linspace(data[feature].min(), data[feature].max(), num=grid_resolution)
        partial_dependence_ = np.zeros_like(feature_values)

        for model in models:
            model_partial_dependence = []
            for value in feature_values:
                data_copy = data.copy()
                data_copy[feature] = value
                predictions = model.predict(data_copy)
                prediction_mean = np.mean(predictions)
                model_partial_dependence.append(prediction_mean)
            partial_dependence_ += np.array(model_partial_dependence)

        partial_dependence_ /= len(models)

        # partial_dependence value adjustment
        min_value_of_partial_dependence = partial_dependence_.min()
        partial_dependence_ -= min_value_of_partial_dependence

        # Creating a DataFrame for the partial dependence
        partial_dependence_df = pd.DataFrame(
            {feature: feature_values, "partial_dependence": partial_dependence_}
        )

        partial_dependence_df_per_feature.append(partial_dependence_df)

    return partial_dependence_df_per_feature


def generate_partial_dependence_for_gam(models, features, grid_resolution):

    partial_dependence_df_per_feature = []

    for i, term in enumerate(models[0].terms):
        if term.isintercept:
            continue

        feature_values = None
        partial_dependence_ = np.zeros(grid_resolution)
        confidence_interval_lower = np.zeros(grid_resolution)
        confidence_interval_upper = np.zeros(grid_resolution)

        for model in models:
            generated_feature_values = model.generate_X_grid(term=i, n=grid_resolution)
            model_partial_dependence, model_confidence_interval = model.partial_dependence(
                term=i, X=generated_feature_values, width=0.95
            )

            if feature_values is None:
                feature_values = generated_feature_values[:, term.feature]

            partial_dependence_ += model_partial_dependence
            confidence_interval_lower += model_confidence_interval[0]
            confidence_interval_upper += model_confidence_interval[1]

        partial_dependence_ /= len(models)
        confidence_interval_lower /= len(models)
        confidence_interval_upper /= len(models)

        # partial_dependence value adjustment
        min_value_of_partial_dependence = partial_dependence_.min()
        partial_dependence_ -= min_value_of_partial_dependence
        confidence_interval_lower -= min_value_of_partial_dependence
        confidence_interval_upper -= min_value_of_partial_dependence

        # Creating a DataFrame for the partial dependence
        partial_dependence_df = pd.DataFrame(
            {
                features[i]: feature_values,
                "partial_dependence": partial_dependence_,
                "confidence_interval_lower": confidence_interval_lower,
                "confidence_interval_upper": confidence_interval_upper,
            }
        )

        partial_dependence_df_per_feature.append(partial_dependence_df)

    return partial_dependence_df_per_feature


def generate_partial_dependence_for_gbm(models, features, data, grid_resolution):

    partial_dependence_df_per_feature = []

    for feature in features:
        feature_values = None
        partial_dependence_ = np.zeros(grid_resolution)

        for model in models:
            partial_dependence_results = partial_dependence(model, X=data, features=[feature], grid_resolution=grid_resolution)
            model_feature_values = partial_dependence_results['grid_values'][0]
            model_partial_dependence_values = partial_dependence_results['average'][0]

            # This resolves a bug where a feature doesn't contain many unique values and cannot go to grid_resolution
            if np.all(partial_dependence_ == 0) & len(model_partial_dependence_values) < grid_resolution:
                partial_dependence_ = np.zeros_like(model_partial_dependence_values)

            if feature_values is None:
                feature_values = model_feature_values

            partial_dependence_ += model_partial_dependence_values

        #partial_dependence_ /= len(models)

        # Creating a DataFrame for the partial dependence
        partial_dependence_df = pd.DataFrame(
            {
                feature: feature_values,
                "partial_dependence": partial_dependence_
            }
        )

        partial_dependence_df_per_feature.append(partial_dependence_df)

    return partial_dependence_df_per_feature


def writing_partial_dependence_data(partial_dependence_df_per_feature, path):

    data_path = f"{path}/data"
    os.makedirs(data_path, exist_ok=True)

    for i, partial_dependence_df in enumerate(partial_dependence_df_per_feature):
        feature = partial_dependence_df.columns[0]
        partial_dependence_df.to_csv(
            f"{data_path}/partial_dependence_data_for_{feature}.csv", index=False
        )
        logger.info(
            f"Partial dependence data for {feature} written to {data_path}/partial_dependence_data_for_{feature}.csv"
        )

    return None


def generating_partial_plots(
    partial_dependence_df_per_feature,
    data,
    path,
    plot_confidence_interval,
    plot_feature_density,
    plot_feature_density_as_histogram,
    number_of_bins_in_histogram,
):
    logger.info("Generating partial dependence plots")

    plots_path = f"{path}/plots"
    os.makedirs(plots_path, exist_ok=True)

    for i, partial_dependence_df in enumerate(partial_dependence_df_per_feature):

        feature = partial_dependence_df.columns[0]

        plt.figure()
        plt.title(f"Partial Dependence Plot for {feature}")
        plt.xlabel(feature)
        plt.ylabel("Partial Dependence")

        plt.plot(
            partial_dependence_df[feature],
            partial_dependence_df["partial_dependence"],
            label="Partial Dependence",
        )
        if (
            plot_confidence_interval
            and "confidence_interval_lower" in partial_dependence_df.columns
        ):
            plt.plot(
                partial_dependence_df[feature],
                partial_dependence_df["confidence_interval_lower"],
                c="r",
                ls="--",
                label="Confidence Interval",
            )
            plt.plot(
                partial_dependence_df[feature],
                partial_dependence_df["confidence_interval_upper"],
                c="r",
                ls="--",
            )

        if (
            plot_feature_density
            and not plot_feature_density_as_histogram
            and "feature_density" in partial_dependence_df.columns
        ):
            plt.plot(
                partial_dependence_df[feature],
                partial_dependence_df["feature_density"],
                color="grey",
                linestyle="--",
                label="Feature Density",
            )

        if (
            plot_feature_density_as_histogram
            and "feature_density" in partial_dependence_df.columns
        ):
            bin_edges = np.linspace(
                partial_dependence_df[feature].min(),
                partial_dependence_df[feature].max(),
                number_of_bins_in_histogram + 1,
            )
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            binned_density, _ = np.histogram(
                partial_dependence_df[feature],
                bins=bin_edges,
                weights=partial_dependence_df["feature_density"],
            )
            binned_density = (
                binned_density
                / binned_density.max()
                * partial_dependence_df["partial_dependence"].max()
            )

            plt.bar(
                bin_centers,
                binned_density,
                width=(bin_edges[1] - bin_edges[0]) * 0.90,
                color="grey",
                alpha=0.3,
                label="Feature Density",
            )

        plt.legend()
        plt.savefig(f"{plots_path}/partial_dependence_plot_for_{feature}.png")
        plt.close()

        logger.info(
            f"Partial dependence plot for {feature} written to {plots_path}/partial_dependence_plot_for_{feature}.png"
        )
    logger.info("Partial dependence plots generated successfully")
