import config.paths as paths
from shared.model.generate_model import save_models

def saving_models_iron_concentrate_perc_model(best_models, config):
    save_models(
        models=best_models,
        path=paths.Paths.IRON_CONCENTRATE_PERC_MODELS_FOLDER.value,
        path_suffix=config.iron_concentrate_perc.model.model_name,
    )

def saving_models_iron_concentrate_perc_feed_blend_model(best_models, config):
    save_models(
        models=best_models,
        path=paths.Paths.IRON_CONCENTRATE_PERC_FEED_BLEND_MODELS_FOLDER.value,
        path_suffix=config.iron_concentrate_perc.model.model_name,
    )

def saving_models_silica_concentrate_perc_model(best_models, config):
    save_models(
        models=best_models,
        path=paths.Paths.SILICA_CONCENTRATE_PERC_MODELS_FOLDER.value,
        path_suffix=config.silica_concentrate_perc.model.model_name,
    )

def saving_models_silica_concentrate_perc_feed_blend_model(best_models, config):
    save_models(
        models=best_models,
        path=paths.Paths.SILICA_CONCENTRATE_PERC_FEED_BLEND_MODELS_FOLDER.value,
        path_suffix=config.silica_concentrate_perc.model.model_name,
    )
