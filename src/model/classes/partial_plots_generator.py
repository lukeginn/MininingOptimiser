import config.paths as paths
from shared.model.classes.partial_plot_processor import PartialPlotProcessor
from dataclasses import dataclass


@dataclass
class PartialPlotsGenerator:
    model_config: dict

    def run_for_iron_concentrate_perc(self, best_models, data, training_features):
        if self.model_config.iron_concentrate_perc.partial_plots.generate:
            partial_plot_processor = PartialPlotProcessor(
                model_choice=self.model_config.iron_concentrate_perc.model.model_choice,
                models=best_models,
                data=data[training_features],
                features=training_features,
                path=paths.Paths.IRON_CONCENTRATE_PERC_PARTIAL_PLOTS_PATH.value,
                plot_confidence_interval=self.model_config.iron_concentrate_perc.partial_plots.plot_confidence_interval,
                plot_feature_density=self.model_config.iron_concentrate_perc.partial_plots.plot_feature_density,
                plot_feature_density_as_histogram=self.model_config.iron_concentrate_perc.partial_plots.plot_feature_density_as_histogram,
                number_of_bins_in_histogram=self.model_config.iron_concentrate_perc.partial_plots.number_of_bins_in_histogram,
                grid_resolution=self.model_config.iron_concentrate_perc.partial_plots.grid_resolution,
            )
            partial_plot_processor.run()

    def run_for_iron_concentrate_perc_feed_blend(
        self, best_models, data, training_features
    ):
        if self.model_config.iron_concentrate_perc.partial_plots.generate:
            partial_plot_processor = PartialPlotProcessor(
                model_choice=self.model_config.iron_concentrate_perc.model.model_choice,
                models=best_models,
                data=data[training_features],
                features=training_features,
                path=paths.Paths.IRON_CONCENTRATE_PERC_FEED_BLEND_PARTIAL_PLOTS_PATH.value,
                plot_confidence_interval=self.model_config.iron_concentrate_perc.partial_plots.plot_confidence_interval,
                plot_feature_density=self.model_config.iron_concentrate_perc.partial_plots.plot_feature_density,
                plot_feature_density_as_histogram=self.model_config.iron_concentrate_perc.partial_plots.plot_feature_density_as_histogram,
                number_of_bins_in_histogram=self.model_config.iron_concentrate_perc.partial_plots.number_of_bins_in_histogram,
                grid_resolution=self.model_config.iron_concentrate_perc.partial_plots.grid_resolution,
            )
            partial_plot_processor.run()

    def run_for_silica_concentrate_perc(self, best_models, data, training_features):
        if self.model_config.silica_concentrate_perc.partial_plots.generate:
            partial_plot_processor = PartialPlotProcessor(
                model_choice=self.model_config.silica_concentrate_perc.model.model_choice,
                models=best_models,
                data=data[training_features],
                features=training_features,
                path=paths.Paths.SILICA_CONCENTRATE_PERC_PARTIAL_PLOTS_PATH.value,
                plot_confidence_interval=self.model_config.silica_concentrate_perc.partial_plots.plot_confidence_interval,
                plot_feature_density=self.model_config.silica_concentrate_perc.partial_plots.plot_feature_density,
                plot_feature_density_as_histogram=self.model_config.silica_concentrate_perc.partial_plots.plot_feature_density_as_histogram,
                number_of_bins_in_histogram=self.model_config.silica_concentrate_perc.partial_plots.number_of_bins_in_histogram,
                grid_resolution=self.model_config.silica_concentrate_perc.partial_plots.grid_resolution,
            )
            partial_plot_processor.run()

    def run_for_silica_concentrate_perc_feed_blend(
        self, best_models, data, training_features
    ):
        if self.model_config.silica_concentrate_perc.partial_plots.generate:
            partial_plot_processor = PartialPlotProcessor(
                model_choice=self.model_config.silica_concentrate_perc.model.model_choice,
                models=best_models,
                data=data[training_features],
                features=training_features,
                path=paths.Paths.SILICA_CONCENTRATE_PERC_FEED_BLEND_PARTIAL_PLOTS_PATH.value,
                plot_confidence_interval=self.model_config.silica_concentrate_perc.partial_plots.plot_confidence_interval,
                plot_feature_density=self.model_config.silica_concentrate_perc.partial_plots.plot_feature_density,
                plot_feature_density_as_histogram=self.model_config.silica_concentrate_perc.partial_plots.plot_feature_density_as_histogram,
                number_of_bins_in_histogram=self.model_config.silica_concentrate_perc.partial_plots.number_of_bins_in_histogram,
                grid_resolution=self.model_config.silica_concentrate_perc.partial_plots.grid_resolution,
            )
            partial_plot_processor.run()
