import config.paths as paths
from shared.model.generate_partial_plots import generate_partial_plots

def generating_partial_plots_iron_concentrate_perc_model(best_models, data, training_features, config):
    generate_partial_plots(
        model_choice=config.iron_concentrate_perc.model.model_choice,
        models=best_models,
        data=data[training_features],
        features=training_features,
        path=paths.Paths.IRON_CONCENTRATE_PERC_PARTIAL_PLOTS_PATH.value,
        plot_confidence_interval=config.iron_concentrate_perc.partial_plots.plot_confidence_interval,
        plot_feature_density=config.iron_concentrate_perc.partial_plots.plot_feature_density,
        plot_feature_density_as_histogram=config.iron_concentrate_perc.partial_plots.plot_feature_density_as_histogram,
        number_of_bins_in_histogram=config.iron_concentrate_perc.partial_plots.number_of_bins_in_histogram,
        grid_resolution=config.iron_concentrate_perc.partial_plots.grid_resolution,
    )

def generating_partial_plots_iron_concentrate_perc_feed_blend_model(best_models, data, training_features, config):
    generate_partial_plots(
        model_choice=config.iron_concentrate_perc.model.model_choice,
        models=best_models,
        data=data[training_features],
        features=training_features,
        path=paths.Paths.IRON_CONCENTRATE_PERC_FEED_BLEND_PARTIAL_PLOTS_PATH.value,
        plot_confidence_interval=config.iron_concentrate_perc.partial_plots.plot_confidence_interval,
        plot_feature_density=config.iron_concentrate_perc.partial_plots.plot_feature_density,
        plot_feature_density_as_histogram=config.iron_concentrate_perc.partial_plots.plot_feature_density_as_histogram,
        number_of_bins_in_histogram=config.iron_concentrate_perc.partial_plots.number_of_bins_in_histogram,
        grid_resolution=config.iron_concentrate_perc.partial_plots.grid_resolution,
    )

def generating_partial_plots_silica_concentrate_perc_model(best_models, data, training_features, config):
    generate_partial_plots(
        model_choice=config.silica_concentrate_perc.model.model_choice,
        models=best_models,
        data=data[training_features],
        features=training_features,
        path=paths.Paths.SILICA_CONCENTRATE_PERC_PARTIAL_PLOTS_PATH.value,
        plot_confidence_interval=config.silica_concentrate_perc.partial_plots.plot_confidence_interval,
        plot_feature_density=config.silica_concentrate_perc.partial_plots.plot_feature_density,
        plot_feature_density_as_histogram=config.silica_concentrate_perc.partial_plots.plot_feature_density_as_histogram,
        number_of_bins_in_histogram=config.silica_concentrate_perc.partial_plots.number_of_bins_in_histogram,
        grid_resolution=config.silica_concentrate_perc.partial_plots.grid_resolution,
    )

def generating_partial_plots_silica_concentrate_perc_feed_blend_model(best_models, data, training_features, config):
    generate_partial_plots(
        model_choice=config.silica_concentrate_perc.model.model_choice,
        models=best_models,
        data=data[training_features],
        features=training_features,
        path=paths.Paths.SILICA_CONCENTRATE_PERC_FEED_BLEND_PARTIAL_PLOTS_PATH.value,
        plot_confidence_interval=config.silica_concentrate_perc.partial_plots.plot_confidence_interval,
        plot_feature_density=config.silica_concentrate_perc.partial_plots.plot_feature_density,
        plot_feature_density_as_histogram=config.silica_concentrate_perc.partial_plots.plot_feature_density_as_histogram,
        number_of_bins_in_histogram=config.silica_concentrate_perc.partial_plots.number_of_bins_in_histogram,
        grid_resolution=config.silica_concentrate_perc.partial_plots.grid_resolution,
    )
