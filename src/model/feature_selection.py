from shared.model.feature_selection import feature_selection

def run_feature_selection_iron_concentrate_perc_model(data, config):
    if config.iron_concentrate_perc.model.feature_selection.run:
        (
            training_features,
            training_features_per_method,
            univariable_feature_importance_from_feature_selection,
            feature_importance_from_feature_selection,
        ) = feature_selection(
            data=data,
            target_feature=config.iron_concentrate_perc.model.target,
            filter_low_variance=config.iron_concentrate_perc.model.feature_selection.filter_low_variance.run,
            filter_univariable_feature_importance=config.iron_concentrate_perc.model.feature_selection.filter_univariable_feature_importance.run,
            filter_feature_importance=config.iron_concentrate_perc.model.feature_selection.filter_feature_importance.run,
            low_variance_threshold=config.iron_concentrate_perc.model.feature_selection.filter_low_variance.threshold,
            univariable_feature_importance_method=config.iron_concentrate_perc.model.univariable_feature_importance.method,
            univariable_feature_importance_threshold=config.iron_concentrate_perc.model.feature_selection.filter_univariable_feature_importance.threshold,
            feature_importance_model_choice=config.iron_concentrate_perc.model.model_choice,
            feature_importance_param_grid=config.iron_concentrate_perc.model.hyperparameters,
            feature_importance_threshold=config.iron_concentrate_perc.model.feature_selection.filter_feature_importance.threshold,
            training_features=config.iron_concentrate_perc.model.training_features,
        )
    else:
        training_features = config.iron_concentrate_perc.model.training_features
        training_features_per_method = None

    return training_features, training_features_per_method

def run_feature_selection_iron_concentrate_perc_feed_blend_model(data, config):
    if config.iron_concentrate_perc.model.feature_selection.run:
        (
            training_features,
            training_features_per_method,
            univariable_feature_importance_from_feature_selection,
            feature_importance_from_feature_selection,
        ) = feature_selection(
            data=data,
            target_feature=config.iron_concentrate_perc.model.target,
            filter_low_variance=config.iron_concentrate_perc.model.feature_selection.filter_low_variance.run,
            filter_univariable_feature_importance=config.iron_concentrate_perc.model.feature_selection.filter_univariable_feature_importance.run,
            filter_feature_importance=config.iron_concentrate_perc.model.feature_selection.filter_feature_importance.run,
            low_variance_threshold=config.iron_concentrate_perc.model.feature_selection.filter_low_variance.threshold,
            univariable_feature_importance_method=config.iron_concentrate_perc.model.univariable_feature_importance.method,
            univariable_feature_importance_threshold=config.iron_concentrate_perc.model.feature_selection.filter_univariable_feature_importance.threshold,
            feature_importance_model_choice=config.iron_concentrate_perc.model.model_choice,
            feature_importance_param_grid=config.iron_concentrate_perc.model.hyperparameters,
            feature_importance_threshold=config.iron_concentrate_perc.model.feature_selection.filter_feature_importance.threshold,
            training_features=config.iron_concentrate_perc.model.feed_blend_training_features,
        )
    else:
        training_features = config.iron_concentrate_perc.model.feed_blend_training_features
        training_features_per_method = None

    return training_features, training_features_per_method

def run_feature_selection_silica_concentrate_perc_model(data, config):
    if config.silica_concentrate_perc.model.feature_selection.run:
        (
            training_features,
            training_features_per_method,
            univariable_feature_importance_from_feature_selection,
            feature_importance_from_feature_selection,
        ) = feature_selection(
            data=data,
            target_feature=config.silica_concentrate_perc.model.target,
            filter_low_variance=config.silica_concentrate_perc.model.feature_selection.filter_low_variance.run,
            filter_univariable_feature_importance=config.silica_concentrate_perc.model.feature_selection.filter_univariable_feature_importance.run,
            filter_feature_importance=config.silica_concentrate_perc.model.feature_selection.filter_feature_importance.run,
            low_variance_threshold=config.silica_concentrate_perc.model.feature_selection.filter_low_variance.threshold,
            univariable_feature_importance_method=config.silica_concentrate_perc.model.univariable_feature_importance.method,
            univariable_feature_importance_threshold=config.silica_concentrate_perc.model.feature_selection.filter_univariable_feature_importance.threshold,
            feature_importance_model_choice=config.silica_concentrate_perc.model.model_choice,
            feature_importance_param_grid=config.silica_concentrate_perc.model.hyperparameters,
            feature_importance_threshold=config.silica_concentrate_perc.model.feature_selection.filter_feature_importance.threshold,
            training_features=config.silica_concentrate_perc.model.training_features,
        )
    else:
        training_features = config.silica_concentrate_perc.model.training_features
        training_features_per_method = None

    return training_features, training_features_per_method

def run_feature_selection_silica_concentrate_perc_feed_blend_model(data, config):
    if config.silica_concentrate_perc.model.feature_selection.run:
        (
            training_features,
            training_features_per_method,
            univariable_feature_importance_from_feature_selection,
            feature_importance_from_feature_selection,
        ) = feature_selection(
            data=data,
            target_feature=config.silica_concentrate_perc.model.target,
            filter_low_variance=config.silica_concentrate_perc.model.feature_selection.filter_low_variance.run,
            filter_univariable_feature_importance=config.silica_concentrate_perc.model.feature_selection.filter_univariable_feature_importance.run,
            filter_feature_importance=config.silica_concentrate_perc.model.feature_selection.filter_feature_importance.run,
            low_variance_threshold=config.silica_concentrate_perc.model.feature_selection.filter_low_variance.threshold,
            univariable_feature_importance_method=config.silica_concentrate_perc.model.univariable_feature_importance.method,
            univariable_feature_importance_threshold=config.silica_concentrate_perc.model.feature_selection.filter_univariable_feature_importance.threshold,
            feature_importance_model_choice=config.silica_concentrate_perc.model.model_choice,
            feature_importance_param_grid=config.silica_concentrate_perc.model.hyperparameters,
            feature_importance_threshold=config.silica_concentrate_perc.model.feature_selection.filter_feature_importance.threshold,
            training_features=config.silica_concentrate_perc.model.feed_blend_training_features,
        )
    else:
        training_features = config.silica_concentrate_perc.model.feed_blend_training_features
        training_features_per_method = None

    return training_features, training_features_per_method