from shared.model.feature_selection import feature_selection

class FeatureSelection:
    def __init__(self, model_config):
        self.model_config = model_config

    def run_for_iron_concentrate_perc(self, data):
        if self.model_config.iron_concentrate_perc.model.feature_selection.run:
            (
                training_features,
                training_features_per_method,
                univariable_feature_importance_from_feature_selection,
                feature_importance_from_feature_selection,
            ) = feature_selection(
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
        else:
            training_features = self.model_config.iron_concentrate_perc.model.training_features
            training_features_per_method = None

        return training_features, training_features_per_method

    def run_for_iron_concentrate_perc_feed_blend(self, data):
        if self.model_config.iron_concentrate_perc.model.feature_selection.run:
            (
                training_features,
                training_features_per_method,
                univariable_feature_importance_from_feature_selection,
                feature_importance_from_feature_selection,
            ) = feature_selection(
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
        else:
            training_features = self.model_config.iron_concentrate_perc.model.feed_blend_training_features
            training_features_per_method = None

        return training_features, training_features_per_method

    def run_for_silica_concentrate_perc(self, data):
        if self.model_config.silica_concentrate_perc.model.feature_selection.run:
            (
                training_features,
                training_features_per_method,
                univariable_feature_importance_from_feature_selection,
                feature_importance_from_feature_selection,
            ) = feature_selection(
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
        else:
            training_features = self.model_config.silica_concentrate_perc.model.training_features
            training_features_per_method = None

        return training_features, training_features_per_method

    def run_for_silica_concentrate_perc_feed_blend(self, data):
        if self.model_config.silica_concentrate_perc.model.feature_selection.run:
            (
                training_features,
                training_features_per_method,
                univariable_feature_importance_from_feature_selection,
                feature_importance_from_feature_selection,
            ) = feature_selection(
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
        else:
            training_features = self.model_config.silica_concentrate_perc.model.feed_blend_training_features
            training_features_per_method = None

        return training_features, training_features_per_method