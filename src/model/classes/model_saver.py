import config.paths as paths
from shared.model.generate_model import save_models
from dataclasses import dataclass


@dataclass
class ModelSaver:
    model_config: dict

    def run_for_iron_concentrate_perc(self, best_models):
        save_models(
            models=best_models,
            path=paths.Paths.IRON_CONCENTRATE_PERC_MODELS_FOLDER.value,
            path_suffix=self.model_config.iron_concentrate_perc.model.model_name,
        )

    def run_for_iron_concentrate_perc_feed_blend(self, best_models):
        save_models(
            models=best_models,
            path=paths.Paths.IRON_CONCENTRATE_PERC_FEED_BLEND_MODELS_FOLDER.value,
            path_suffix=self.model_config.iron_concentrate_perc.model.model_name,
        )

    def run_for_silica_concentrate_perc(self, best_models):
        save_models(
            models=best_models,
            path=paths.Paths.SILICA_CONCENTRATE_PERC_MODELS_FOLDER.value,
            path_suffix=self.model_config.silica_concentrate_perc.model.model_name,
        )

    def run_for_silica_concentrate_perc_feed_blend(self, best_models):
        save_models(
            models=best_models,
            path=paths.Paths.SILICA_CONCENTRATE_PERC_FEED_BLEND_MODELS_FOLDER.value,
            path_suffix=self.model_config.silica_concentrate_perc.model.model_name,
        )
