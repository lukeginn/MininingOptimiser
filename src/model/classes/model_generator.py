import config.paths as paths
from shared.model.classes.model_processor import ModelProcessor
from dataclasses import dataclass


@dataclass
class ModelGenerator:
    model_config: dict

    def run_for_iron_concentrate_perc(self, data, training_features):
        model_processor = ModelProcessor(
            data=data,
            target_feature=self.model_config.iron_concentrate_perc.model.target,
            training_features=training_features,
            model_choice=self.model_config.iron_concentrate_perc.model.model_choice,
            param_grid=self.model_config.iron_concentrate_perc.model.hyperparameters,
            metric=self.model_config.iron_concentrate_perc.model.metric,
            generate_feature_importance=self.model_config.iron_concentrate_perc.model.generate_feature_importance,
            feature_importance_path=paths.Paths.IRON_CONCENTRATE_PERC_FEATURE_IMPORTANCE_FILE.value,
            evaluation_results_path=paths.Paths.IRON_CONCENTRATE_PERC_MODEL_EVALUATION_RESULTS_FILE.value,
            random_state=self.model_config.iron_concentrate_perc.model.random_state,
            n_models=self.model_config.iron_concentrate_perc.model.number_of_models,
            path=paths.Paths.IRON_CONCENTRATE_PERC_MODEL_EVALUATION_SCATTER_PLOT.value,
        )
        best_models, best_params, best_rmse, feature_importance = model_processor.run()
        return best_models

    def run_for_iron_concentrate_perc_feed_blend(self, data, training_features):
        model_processor = ModelProcessor(
            data=data,
            target_feature=self.model_config.iron_concentrate_perc.model.target,
            training_features=training_features,
            model_choice=self.model_config.iron_concentrate_perc.model.model_choice,
            param_grid=self.model_config.iron_concentrate_perc.model.hyperparameters,
            metric=self.model_config.iron_concentrate_perc.model.metric,
            generate_feature_importance=self.model_config.iron_concentrate_perc.model.generate_feature_importance,
            feature_importance_path=paths.Paths.IRON_CONCENTRATE_PERC_FEED_BLEND_FEATURE_IMPORTANCE_FILE.value,
            evaluation_results_path=paths.Paths.IRON_CONCENTRATE_PERC_FEED_BLEND_MODEL_EVALUATION_RESULTS_FILE.value,
            random_state=self.model_config.iron_concentrate_perc.model.random_state,
            n_models=self.model_config.iron_concentrate_perc.model.number_of_models,
            path=paths.Paths.IRON_CONCENTRATE_PERC_FEED_BLEND_MODEL_EVALUATION_SCATTER_PLOT.value,
        )
        best_models, best_params, best_rmse, feature_importance = model_processor.run()
        return best_models

    def run_for_silica_concentrate_perc(self, data, training_features):
        model_processor = ModelProcessor(
            data=data,
            target_feature=self.model_config.silica_concentrate_perc.model.target,
            training_features=training_features,
            model_choice=self.model_config.silica_concentrate_perc.model.model_choice,
            param_grid=self.model_config.silica_concentrate_perc.model.hyperparameters,
            metric=self.model_config.silica_concentrate_perc.model.metric,
            generate_feature_importance=self.model_config.silica_concentrate_perc.model.generate_feature_importance,
            feature_importance_path=paths.Paths.SILICA_CONCENTRATE_PERC_FEATURE_IMPORTANCE_FILE.value,
            evaluation_results_path=paths.Paths.SILICA_CONCENTRATE_PERC_MODEL_EVALUATION_RESULTS_FILE.value,
            random_state=self.model_config.silica_concentrate_perc.model.random_state,
            n_models=self.model_config.silica_concentrate_perc.model.number_of_models,
            path=paths.Paths.SILICA_CONCENTRATE_PERC_MODEL_EVALUATION_SCATTER_PLOT.value,
        )
        best_models, best_params, best_rmse, feature_importance = model_processor.run()
        return best_models

    def run_for_silica_concentrate_perc_feed_blend(self, data, training_features):
        model_processor = ModelProcessor(
            data=data,
            target_feature=self.model_config.silica_concentrate_perc.model.target,
            training_features=training_features,
            model_choice=self.model_config.silica_concentrate_perc.model.model_choice,
            param_grid=self.model_config.silica_concentrate_perc.model.hyperparameters,
            metric=self.model_config.silica_concentrate_perc.model.metric,
            generate_feature_importance=self.model_config.silica_concentrate_perc.model.generate_feature_importance,
            feature_importance_path=paths.Paths.SILICA_CONCENTRATE_PERC_FEED_BLEND_FEATURE_IMPORTANCE_FILE.value,
            evaluation_results_path=paths.Paths.SILICA_CONCENTRATE_PERC_FEED_BLEND_MODEL_EVALUATION_RESULTS_FILE.value,
            random_state=self.model_config.silica_concentrate_perc.model.random_state,
            n_models=self.model_config.silica_concentrate_perc.model.number_of_models,
            path=paths.Paths.SILICA_CONCENTRATE_PERC_FEED_BLEND_MODEL_EVALUATION_SCATTER_PLOT.value,
        )
        best_models, best_params, best_rmse, feature_importance = model_processor.run()
        return best_models
