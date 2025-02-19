import config.paths as paths
from shared.model.generate_model import generate_model

def generating_model_iron_concentrate_perc_model(data, training_features, config):
    best_models, best_params, best_rmse, feature_importance = generate_model(
        data=data,
        target_feature=config.iron_concentrate_perc.model.target,
        training_features=training_features,
        model_choice=config.iron_concentrate_perc.model.model_choice,
        param_grid=config.iron_concentrate_perc.model.hyperparameters,
        metric=config.iron_concentrate_perc.model.metric,
        generate_feature_importance=config.iron_concentrate_perc.model.generate_feature_importance,
        feature_importance_path=paths.Paths.IRON_CONCENTRATE_PERC_FEATURE_IMPORTANCE_FILE.value,
        evaluation_results_path=paths.Paths.IRON_CONCENTRATE_PERC_MODEL_EVALUATION_RESULTS_FILE.value,
        random_state=config.iron_concentrate_perc.model.random_state,
        n_models=config.iron_concentrate_perc.model.number_of_models,
        path=paths.Paths.IRON_CONCENTRATE_PERC_MODEL_EVALUATION_SCATTER_PLOT.value
    )
    return best_models, best_params, best_rmse, feature_importance

def generating_model_iron_concentrate_perc_feed_blend_model(data, training_features, config):
    best_models, best_params, best_rmse, feature_importance = generate_model(
        data=data,
        target_feature=config.iron_concentrate_perc.model.target,
        training_features=training_features,
        model_choice=config.iron_concentrate_perc.model.model_choice,
        param_grid=config.iron_concentrate_perc.model.hyperparameters,
        metric=config.iron_concentrate_perc.model.metric,
        generate_feature_importance=config.iron_concentrate_perc.model.generate_feature_importance,
        feature_importance_path=paths.Paths.IRON_CONCENTRATE_PERC_FEED_BLEND_FEATURE_IMPORTANCE_FILE.value,
        evaluation_results_path=paths.Paths.IRON_CONCENTRATE_PERC_FEED_BLEND_MODEL_EVALUATION_RESULTS_FILE.value,
        random_state=config.iron_concentrate_perc.model.random_state,
        n_models=config.iron_concentrate_perc.model.number_of_models,
        path=paths.Paths.IRON_CONCENTRATE_PERC_FEED_BLEND_MODEL_EVALUATION_SCATTER_PLOT.value
    )
    return best_models, best_params, best_rmse, feature_importance

def generating_model_silica_concentrate_perc_model(data, training_features, config):
    best_models, best_params, best_rmse, feature_importance = generate_model(
        data=data,
        target_feature=config.silica_concentrate_perc.model.target,
        training_features=training_features,
        model_choice=config.silica_concentrate_perc.model.model_choice,
        param_grid=config.silica_concentrate_perc.model.hyperparameters,
        metric=config.silica_concentrate_perc.model.metric,
        generate_feature_importance=config.silica_concentrate_perc.model.generate_feature_importance,
        feature_importance_path=paths.Paths.SILICA_CONCENTRATE_PERC_FEATURE_IMPORTANCE_FILE.value,
        evaluation_results_path=paths.Paths.SILICA_CONCENTRATE_PERC_MODEL_EVALUATION_RESULTS_FILE.value,
        random_state=config.silica_concentrate_perc.model.random_state,
        n_models=config.silica_concentrate_perc.model.number_of_models,
        path=paths.Paths.SILICA_CONCENTRATE_PERC_MODEL_EVALUATION_SCATTER_PLOT.value
    )
    return best_models, best_params, best_rmse, feature_importance

def generating_model_silica_concentrate_perc_feed_blend_model(data, training_features, config):
    best_models, best_params, best_rmse, feature_importance = generate_model(
        data=data,
        target_feature=config.silica_concentrate_perc.model.target,
        training_features=training_features,
        model_choice=config.silica_concentrate_perc.model.model_choice,
        param_grid=config.silica_concentrate_perc.model.hyperparameters,
        metric=config.silica_concentrate_perc.model.metric,
        generate_feature_importance=config.silica_concentrate_perc.model.generate_feature_importance,
        feature_importance_path=paths.Paths.SILICA_CONCENTRATE_PERC_FEED_BLEND_FEATURE_IMPORTANCE_FILE.value,
        evaluation_results_path=paths.Paths.SILICA_CONCENTRATE_PERC_FEED_BLEND_MODEL_EVALUATION_RESULTS_FILE.value,
        random_state=config.silica_concentrate_perc.model.random_state,
        n_models=config.silica_concentrate_perc.model.number_of_models,
        path=paths.Paths.SILICA_CONCENTRATE_PERC_FEED_BLEND_MODEL_EVALUATION_SCATTER_PLOT.value
    )
    return best_models, best_params, best_rmse, feature_importance
