from shared.model.classes.feature_selection_processor import FeatureSelectorProcessor
from dataclasses import dataclass


@dataclass
class FeatureSelection:
    model_config: dict

    def run_for_iron_concentrate_perc(self, data):
        if self.model_config.iron_concentrate_perc.model.feature_selection.run:
            feature_selection_processor = FeatureSelectorProcessor(
                data=data,
                target_feature=self.model_config.iron_concentrate_perc.model.target,
                filter_low_variance=self.model_config.iron_concentrate_perc.model.feature_selection.filter_low_variance.run,
                filter_univariable_feature_importance=self.model_config.iron_concentrate_perc.model.feature_selection.filter_univariable_feature_importance.run,
                filter_feature_importance=self.model_config.iron_concentrate_perc.model.feature_selection.filter_feature_importance.run,
                low_variance_threshold=self.model_config.iron_concentrate_perc.model.feature_selection.filter_low_variance.threshold,
                univariable_feature_importance_method=self.model_config.iron_concentrate_perc.model.univariable_feature_importance.method,
                univariable_feature_importance_threshold=self.model_config.iron_concentrate_perc.model.feature_selection.filter_univariable_feature_importance.threshold,
                feature_importance_model_choice=self.model_config.iron_concentrate_perc.model.model_choice,
                feature_importance_param_grid=self.model_config.iron_concentrate_perc.model.hyperparameters,
                feature_importance_threshold=self.model_config.iron_concentrate_perc.model.feature_selection.filter_feature_importance.threshold,
                training_features=self.model_config.iron_concentrate_perc.model.training_features,
            )
            feature_selection_results = feature_selection_processor.run()
            training_features = feature_selection_results["training_features"]
            training_features_per_method = feature_selection_results["training_features_per_method"]
        else:
            training_features = (
                self.model_config.iron_concentrate_perc.model.training_features
            )
            training_features_per_method = None

        return training_features, training_features_per_method

    def run_for_iron_concentrate_perc_feed_blend(self, data):
        if self.model_config.iron_concentrate_perc.model.feature_selection.run:
            feature_selection_processor = FeatureSelectorProcessor(
                data=data,
                target_feature=self.model_config.iron_concentrate_perc.model.target,
                filter_low_variance=self.model_config.iron_concentrate_perc.model.feature_selection.filter_low_variance.run,
                filter_univariable_feature_importance=self.model_config.iron_concentrate_perc.model.feature_selection.filter_univariable_feature_importance.run,
                filter_feature_importance=self.model_config.iron_concentrate_perc.model.feature_selection.filter_feature_importance.run,
                low_variance_threshold=self.model_config.iron_concentrate_perc.model.feature_selection.filter_low_variance.threshold,
                univariable_feature_importance_method=self.model_config.iron_concentrate_perc.model.univariable_feature_importance.method,
                univariable_feature_importance_threshold=self.model_config.iron_concentrate_perc.model.feature_selection.filter_univariable_feature_importance.threshold,
                feature_importance_model_choice=self.model_config.iron_concentrate_perc.model.model_choice,
                feature_importance_param_grid=self.model_config.iron_concentrate_perc.model.hyperparameters,
                feature_importance_threshold=self.model_config.iron_concentrate_perc.model.feature_selection.filter_feature_importance.threshold,
                training_features=self.model_config.iron_concentrate_perc.model.feed_blend_training_features,
            )
            feature_selection_results = feature_selection_processor.run()
            training_features = feature_selection_results["training_features"]
            training_features_per_method = feature_selection_results["training_features_per_method"]
        else:
            training_features = (
                self.model_config.iron_concentrate_perc.model.feed_blend_training_features
            )
            training_features_per_method = None

        return training_features, training_features_per_method

    def run_for_silica_concentrate_perc(self, data):
        if self.model_config.silica_concentrate_perc.model.feature_selection.run:
            feature_selection_processor = FeatureSelectorProcessor(
                data=data,
                target_feature=self.model_config.silica_concentrate_perc.model.target,
                filter_low_variance=self.model_config.silica_concentrate_perc.model.feature_selection.filter_low_variance.run,
                filter_univariable_feature_importance=self.model_config.silica_concentrate_perc.model.feature_selection.filter_univariable_feature_importance.run,
                filter_feature_importance=self.model_config.silica_concentrate_perc.model.feature_selection.filter_feature_importance.run,
                low_variance_threshold=self.model_config.silica_concentrate_perc.model.feature_selection.filter_low_variance.threshold,
                univariable_feature_importance_method=self.model_config.silica_concentrate_perc.model.univariable_feature_importance.method,
                univariable_feature_importance_threshold=self.model_config.silica_concentrate_perc.model.feature_selection.filter_univariable_feature_importance.threshold,
                feature_importance_model_choice=self.model_config.silica_concentrate_perc.model.model_choice,
                feature_importance_param_grid=self.model_config.silica_concentrate_perc.model.hyperparameters,
                feature_importance_threshold=self.model_config.silica_concentrate_perc.model.feature_selection.filter_feature_importance.threshold,
                training_features=self.model_config.silica_concentrate_perc.model.training_features,
            )
            feature_selection_results = feature_selection_processor.run()
            training_features = feature_selection_results["training_features"]
            training_features_per_method = feature_selection_results["training_features_per_method"]
        else:
            training_features = (
                self.model_config.silica_concentrate_perc.model.training_features
            )
            training_features_per_method = None

        return training_features, training_features_per_method

    def run_for_silica_concentrate_perc_feed_blend(self, data):
        if self.model_config.silica_concentrate_perc.model.feature_selection.run:
            feature_selection_processor = FeatureSelectorProcessor(
                data=data,
                target_feature=self.model_config.silica_concentrate_perc.model.target,
                filter_low_variance=self.model_config.silica_concentrate_perc.model.feature_selection.filter_low_variance.run,
                filter_univariable_feature_importance=self.model_config.silica_concentrate_perc.model.feature_selection.filter_univariable_feature_importance.run,
                filter_feature_importance=self.model_config.silica_concentrate_perc.model.feature_selection.filter_feature_importance.run,
                low_variance_threshold=self.model_config.silica_concentrate_perc.model.feature_selection.filter_low_variance.threshold,
                univariable_feature_importance_method=self.model_config.silica_concentrate_perc.model.univariable_feature_importance.method,
                univariable_feature_importance_threshold=self.model_config.silica_concentrate_perc.model.feature_selection.filter_univariable_feature_importance.threshold,
                feature_importance_model_choice=self.model_config.silica_concentrate_perc.model.model_choice,
                feature_importance_param_grid=self.model_config.silica_concentrate_perc.model.hyperparameters,
                feature_importance_threshold=self.model_config.silica_concentrate_perc.model.feature_selection.filter_feature_importance.threshold,
                training_features=self.model_config.silica_concentrate_perc.model.feed_blend_training_features,
            )
            feature_selection_results = feature_selection_processor.run()
            training_features = feature_selection_results["training_features"]
            training_features_per_method = feature_selection_results["training_features_per_method"]
        else:
            training_features = (
                self.model_config.silica_concentrate_perc.model.feed_blend_training_features
            )
            training_features_per_method = None

        return training_features, training_features_per_method
