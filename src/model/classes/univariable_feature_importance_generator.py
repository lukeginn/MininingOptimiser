import config.paths as paths
from shared.model.univariable_feature_importance import generate_univariable_feature_importance

class UnivariableFeatureImportanceGenerator:
    def __init__(self, config):
        self.config = config

    def run_for_iron_concentrate_perc(self, data):
        if self.config.iron_concentrate_perc.model.univariable_feature_importance.run:
            univariable_feature_importance = generate_univariable_feature_importance(
                data=data,
                target_feature=self.config.iron_concentrate_perc.model.target,
                method=self.config.iron_concentrate_perc.model.univariable_feature_importance.method,
                path=paths.Paths.IRON_CONCENTRATE_PERC_UNIVARIABLE_FEATURE_IMPORTANCE_PATH.value,
            )
            return univariable_feature_importance

    def run_for_silica_concentrate_perc(self, data):
        if self.config.silica_concentrate_perc.model.univariable_feature_importance.run:
            univariable_feature_importance = generate_univariable_feature_importance(
                data=data,
                target_feature=self.config.silica_concentrate_perc.model.target,
                method=self.config.silica_concentrate_perc.model.univariable_feature_importance.method,
                path=paths.Paths.SILICA_CONCENTRATE_PERC_UNIVARIABLE_FEATURE_IMPORTANCE_PATH.value,
            )
            return univariable_feature_importance
