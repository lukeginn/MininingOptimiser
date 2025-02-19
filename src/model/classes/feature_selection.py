from shared.model.feature_selection import feature_selection

class FeatureSelection:
    def __init__(self, config):
        self.config = config

    def run_for_iron_concentrate_perc(self, data):
        if self.config.iron_concentrate_perc.model.feature_selection.run:
            (
                training_features,
                training_features_per_method,
                univariable_feature_importance_from_feature_selection,
                feature_importance_from_feature_selection,
            ) = feature_selection(
                data=data,
                target_feature=self.config.iron_concentrate_perc.model.target,
                filter_low_variance=self.config.iron_concentrate_perc.model.feature_selection.filter_low_variance.run,
                filter_univariable_feature_importance=self.config.iron_concentrate_perc.model.feature_selection.filter_univariable_feature_importance.run,
                filter_feature_importance=self.config.iron_concentrate_perc.model.feature_selection.filter_feature_importance.run,
                low_variance_threshold=self.config.iron_concentrate_perc.model.feature_selection.filter_low_variance.threshold,
                univariable_feature_importance_method=self.config.iron_concentrate_perc.model.univariable_feature_importance.method,
                univariable_feature_importance_threshold=self.config.iron_concentrate_perc.model.feature_selection.filter_univariable_feature_importance.threshold,
                feature_importance_model_choice=self.config.iron_concentrate_perc.model.model_choice,
                feature_importance_param_grid=self.config.iron_concentrate_perc.model.hyperparameters,
                feature_importance_threshold=self.config.iron_concentrate_perc.model.feature_selection.filter_feature_importance.threshold,
                training_features=self.config.iron_concentrate_perc.model.training_features,
            )
        else:
            training_features = self.config.iron_concentrate_perc.model.training_features
            training_features_per_method = None

        return training_features, training_features_per_method

    def run_for_iron_concentrate_perc_feed_blend(self, data):
        if self.config.iron_concentrate_perc.model.feature_selection.run:
            (
                training_features,
                training_features_per_method,
                univariable_feature_importance_from_feature_selection,
                feature_importance_from_feature_selection,
            ) = feature_selection(
                data=data,
                target_feature=self.config.iron_concentrate_perc.model.target,
                filter_low_variance=self.config.iron_concentrate_perc.model.feature_selection.filter_low_variance.run,
                filter_univariable_feature_importance=self.config.iron_concentrate_perc.model.feature_selection.filter_univariable_feature_importance.run,
                filter_feature_importance=self.config.iron_concentrate_perc.model.feature_selection.filter_feature_importance.run,
                low_variance_threshold=self.config.iron_concentrate_perc.model.feature_selection.filter_low_variance.threshold,
                univariable_feature_importance_method=self.config.iron_concentrate_perc.model.univariable_feature_importance.method,
                univariable_feature_importance_threshold=self.config.iron_concentrate_perc.model.feature_selection.filter_univariable_feature_importance.threshold,
                feature_importance_model_choice=self.config.iron_concentrate_perc.model.model_choice,
                feature_importance_param_grid=self.config.iron_concentrate_perc.model.hyperparameters,
                feature_importance_threshold=self.config.iron_concentrate_perc.model.feature_selection.filter_feature_importance.threshold,
                training_features=self.config.iron_concentrate_perc.model.feed_blend_training_features,
            )
        else:
            training_features = self.config.iron_concentrate_perc.model.feed_blend_training_features
            training_features_per_method = None

        return training_features, training_features_per_method

    def run_for_silica_concentrate_perc(self, data):
        if self.config.silica_concentrate_perc.model.feature_selection.run:
            (
                training_features,
                training_features_per_method,
                univariable_feature_importance_from_feature_selection,
                feature_importance_from_feature_selection,
            ) = feature_selection(
                data=data,
                target_feature=self.config.silica_concentrate_perc.model.target,
                filter_low_variance=self.config.silica_concentrate_perc.model.feature_selection.filter_low_variance.run,
                filter_univariable_feature_importance=self.config.silica_concentrate_perc.model.feature_selection.filter_univariable_feature_importance.run,
                filter_feature_importance=self.config.silica_concentrate_perc.model.feature_selection.filter_feature_importance.run,
                low_variance_threshold=self.config.silica_concentrate_perc.model.feature_selection.filter_low_variance.threshold,
                univariable_feature_importance_method=self.config.silica_concentrate_perc.model.univariable_feature_importance.method,
                univariable_feature_importance_threshold=self.config.silica_concentrate_perc.model.feature_selection.filter_univariable_feature_importance.threshold,
                feature_importance_model_choice=self.config.silica_concentrate_perc.model.model_choice,
                feature_importance_param_grid=self.config.silica_concentrate_perc.model.hyperparameters,
                feature_importance_threshold=self.config.silica_concentrate_perc.model.feature_selection.filter_feature_importance.threshold,
                training_features=self.config.silica_concentrate_perc.model.training_features,
            )
        else:
            training_features = self.config.silica_concentrate_perc.model.training_features
            training_features_per_method = None

        return training_features, training_features_per_method

    def run_for_silica_concentrate_perc_feed_blend(self, data):
        if self.config.silica_concentrate_perc.model.feature_selection.run:
            (
                training_features,
                training_features_per_method,
                univariable_feature_importance_from_feature_selection,
                feature_importance_from_feature_selection,
            ) = feature_selection(
                data=data,
                target_feature=self.config.silica_concentrate_perc.model.target,
                filter_low_variance=self.config.silica_concentrate_perc.model.feature_selection.filter_low_variance.run,
                filter_univariable_feature_importance=self.config.silica_concentrate_perc.model.feature_selection.filter_univariable_feature_importance.run,
                filter_feature_importance=self.config.silica_concentrate_perc.model.feature_selection.filter_feature_importance.run,
                low_variance_threshold=self.config.silica_concentrate_perc.model.feature_selection.filter_low_variance.threshold,
                univariable_feature_importance_method=self.config.silica_concentrate_perc.model.univariable_feature_importance.method,
                univariable_feature_importance_threshold=self.config.silica_concentrate_perc.model.feature_selection.filter_univariable_feature_importance.threshold,
                feature_importance_model_choice=self.config.silica_concentrate_perc.model.model_choice,
                feature_importance_param_grid=self.config.silica_concentrate_perc.model.hyperparameters,
                feature_importance_threshold=self.config.silica_concentrate_perc.model.feature_selection.filter_feature_importance.threshold,
                training_features=self.config.silica_concentrate_perc.model.feed_blend_training_features,
            )
        else:
            training_features = self.config.silica_concentrate_perc.model.feed_blend_training_features
            training_features_per_method = None

        return training_features, training_features_per_method