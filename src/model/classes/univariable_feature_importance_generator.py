import config.paths as paths
from shared.model.univariable_feature_importance import (
    generate_univariable_feature_importance,
)
from dataclasses import dataclass


@dataclass
class UnivariableFeatureImportanceGenerator:
    model_config: dict

    def run_for_iron_concentrate_perc(self, data):
        if (
            self.model_config.iron_concentrate_perc.model.univariable_feature_importance.run
        ):
            univariable_feature_importance = generate_univariable_feature_importance(
                data=data,
                target_feature=self.model_config.iron_concentrate_perc.model.target,
                method=self.model_config.iron_concentrate_perc.model.univariable_feature_importance.method,
                path=paths.Paths.IRON_CONCENTRATE_PERC_UNIVARIABLE_FEATURE_IMPORTANCE_PATH.value,
            )
            return univariable_feature_importance

    def run_for_silica_concentrate_perc(self, data):
        if (
            self.model_config.silica_concentrate_perc.model.univariable_feature_importance.run
        ):
            univariable_feature_importance = generate_univariable_feature_importance(
                data=data,
                target_feature=self.model_config.silica_concentrate_perc.model.target,
                method=self.model_config.silica_concentrate_perc.model.univariable_feature_importance.method,
                path=paths.Paths.SILICA_CONCENTRATE_PERC_UNIVARIABLE_FEATURE_IMPORTANCE_PATH.value,
            )
            return univariable_feature_importance
