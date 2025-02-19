
import config.paths as paths
from shared.model.univariable_feature_importance import generate_univariable_feature_importance

def generating_univariabe_feature_importance_for_iron_concentrate_perc_model(data, config):
    if config.iron_concentrate_perc.model.univariable_feature_importance.run:
        univariable_feature_importance = generate_univariable_feature_importance(
            data=data,
            target_feature=config.iron_concentrate_perc.model.target,
            method=config.iron_concentrate_perc.model.univariable_feature_importance.method,
            path=paths.Paths.IRON_CONCENTRATE_PERC_UNIVARIABLE_FEATURE_IMPORTANCE_PATH.value,
        )
        return univariable_feature_importance
    
def generating_univariabe_feature_importance_for_silica_concentrate_perc_model(data, config):
    if config.silica_concentrate_perc.model.univariable_feature_importance.run:
        univariable_feature_importance = generate_univariable_feature_importance(
            data=data,
            target_feature=config.silica_concentrate_perc.model.target,
            method=config.silica_concentrate_perc.model.univariable_feature_importance.method,
            path=paths.Paths.SILICA_CONCENTRATE_PERC_UNIVARIABLE_FEATURE_IMPORTANCE_PATH.value,
        )
        return univariable_feature_importance
