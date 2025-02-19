import config.paths as paths
from shared.model.generate_model import generate_model

class ModelGenerator:
    def __init__(self, config):
        self.config = config

    def run_for_iron_concentrate_perc(self, data, training_features):
        best_models, best_params, best_rmse, feature_importance = generate_model(
            data=data,
            target_feature=self.config.iron_concentrate_perc.model.target,
            training_features=training_features,
            model_choice=self.config.iron_concentrate_perc.model.model_choice,
            param_grid=self.config.iron_concentrate_perc.model.hyperparameters,
            metric=self.config.iron_concentrate_perc.model.metric,
            generate_feature_importance=self.config.iron_concentrate_perc.model.generate_feature_importance,
            feature_importance_path=paths.Paths.IRON_CONCENTRATE_PERC_FEATURE_IMPORTANCE_FILE.value,
            evaluation_results_path=paths.Paths.IRON_CONCENTRATE_PERC_MODEL_EVALUATION_RESULTS_FILE.value,
            random_state=self.config.iron_concentrate_perc.model.random_state,
            n_models=self.config.iron_concentrate_perc.model.number_of_models,
            path=paths.Paths.IRON_CONCENTRATE_PERC_MODEL_EVALUATION_SCATTER_PLOT.value
        )
        return best_models, best_params, best_rmse, feature_importance

    def run_for_iron_concentrate_perc_feed_blend(self, data, training_features):
        best_models, best_params, best_rmse, feature_importance = generate_model(
            data=data,
            target_feature=self.config.iron_concentrate_perc.model.target,
            training_features=training_features,
            model_choice=self.config.iron_concentrate_perc.model.model_choice,
            param_grid=self.config.iron_concentrate_perc.model.hyperparameters,
            metric=self.config.iron_concentrate_perc.model.metric,
            generate_feature_importance=self.config.iron_concentrate_perc.model.generate_feature_importance,
            feature_importance_path=paths.Paths.IRON_CONCENTRATE_PERC_FEED_BLEND_FEATURE_IMPORTANCE_FILE.value,
            evaluation_results_path=paths.Paths.IRON_CONCENTRATE_PERC_FEED_BLEND_MODEL_EVALUATION_RESULTS_FILE.value,
            random_state=self.config.iron_concentrate_perc.model.random_state,
            n_models=self.config.iron_concentrate_perc.model.number_of_models,
            path=paths.Paths.IRON_CONCENTRATE_PERC_FEED_BLEND_MODEL_EVALUATION_SCATTER_PLOT.value
        )
        return best_models, best_params, best_rmse, feature_importance

    def run_for_silica_concentrate_perc(self, data, training_features):
        best_models, best_params, best_rmse, feature_importance = generate_model(
            data=data,
            target_feature=self.config.silica_concentrate_perc.model.target,
            training_features=training_features,
            model_choice=self.config.silica_concentrate_perc.model.model_choice,
            param_grid=self.config.silica_concentrate_perc.model.hyperparameters,
            metric=self.config.silica_concentrate_perc.model.metric,
            generate_feature_importance=self.config.silica_concentrate_perc.model.generate_feature_importance,
            feature_importance_path=paths.Paths.SILICA_CONCENTRATE_PERC_FEATURE_IMPORTANCE_FILE.value,
            evaluation_results_path=paths.Paths.SILICA_CONCENTRATE_PERC_MODEL_EVALUATION_RESULTS_FILE.value,
            random_state=self.config.silica_concentrate_perc.model.random_state,
            n_models=self.config.silica_concentrate_perc.model.number_of_models,
            path=paths.Paths.SILICA_CONCENTRATE_PERC_MODEL_EVALUATION_SCATTER_PLOT.value
        )
        return best_models, best_params, best_rmse, feature_importance

    def run_for_silica_concentrate_perc_feed_blend(self, data, training_features):
        best_models, best_params, best_rmse, feature_importance = generate_model(
            data=data,
            target_feature=self.config.silica_concentrate_perc.model.target,
            training_features=training_features,
            model_choice=self.config.silica_concentrate_perc.model.model_choice,
            param_grid=self.config.silica_concentrate_perc.model.hyperparameters,
            metric=self.config.silica_concentrate_perc.model.metric,
            generate_feature_importance=self.config.silica_concentrate_perc.model.generate_feature_importance,
            feature_importance_path=paths.Paths.SILICA_CONCENTRATE_PERC_FEED_BLEND_FEATURE_IMPORTANCE_FILE.value,
            evaluation_results_path=paths.Paths.SILICA_CONCENTRATE_PERC_FEED_BLEND_MODEL_EVALUATION_RESULTS_FILE.value,
            random_state=self.config.silica_concentrate_perc.model.random_state,
            n_models=self.config.silica_concentrate_perc.model.number_of_models,
            path=paths.Paths.SILICA_CONCENTRATE_PERC_FEED_BLEND_MODEL_EVALUATION_SCATTER_PLOT.value
        )
        return best_models, best_params, best_rmse, feature_importance
