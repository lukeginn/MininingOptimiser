import config.paths as paths
from shared.model.generate_model import save_models

class ModelSaver:
    def __init__(self, config):
        self.config = config

    def run_for_iron_concentrate_perc(self, best_models):
        save_models(
            models=best_models,
            path=paths.Paths.IRON_CONCENTRATE_PERC_MODELS_FOLDER.value,
            path_suffix=self.config.iron_concentrate_perc.model.model_name,
        )

    def run_for_iron_concentrate_perc_feed_blend(self, best_models):
        save_models(
            models=best_models,
            path=paths.Paths.IRON_CONCENTRATE_PERC_FEED_BLEND_MODELS_FOLDER.value,
            path_suffix=self.config.iron_concentrate_perc.model.model_name,
        )

    def run_for_silica_concentrate_perc(self, best_models):
        save_models(
            models=best_models,
            path=paths.Paths.SILICA_CONCENTRATE_PERC_MODELS_FOLDER.value,
            path_suffix=self.config.silica_concentrate_perc.model.model_name,
        )

    def run_for_silica_concentrate_perc_feed_blend(self, best_models):
        save_models(
            models=best_models,
            path=paths.Paths.SILICA_CONCENTRATE_PERC_FEED_BLEND_MODELS_FOLDER.value,
            path_suffix=self.config.silica_concentrate_perc.model.model_name,
        )
