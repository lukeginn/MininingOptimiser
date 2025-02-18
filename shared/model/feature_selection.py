import numpy as np
import logging as logger
from shared.model.generate_model import generate_model
from shared.model.univariable_feature_importance import (
    generate_univariable_feature_importance,
)


def feature_selection(
    data,
    target_feature,
    filter_low_variance=True,
    filter_univariable_feature_importance=True,
    filter_feature_importance=False,
    low_variance_threshold=0.01,
    univariable_feature_importance_method="kendall",
    univariable_feature_importance_threshold=1,
    feature_importance_model_choice=None,
    feature_importance_param_grid=None,
    feature_importance_threshold=1,
    training_features=None,
    test_size=0.2,
    random_state=42,
):
    if training_features is None:
        logger.info("No training features provided. Using all numeric features")
        training_features = numeric_only_training_features(
            data=data, target_feature=target_feature
        )

    # Dictionary to store the features selected by each method
    training_features_per_method = dict()

    if filter_low_variance:
        logger.info("Filtering out zero or near zero variance features")
        training_features = filter_out_zero_or_near_zero_variance_features(
            data=data,
            training_features=training_features,
            threshold=low_variance_threshold,
        )
        training_features_per_method["low_variance"] = get_alphabetical_features(
            features=training_features
        )

    if filter_univariable_feature_importance:
        logger.info("Filtering features via univariable feature importance")
        training_features, univariable_feature_importance = (
            select_features_via_univariable_feature_importance(
                data=data,
                target_feature=target_feature,
                training_features=training_features,
                method=univariable_feature_importance_method,
                threshold=univariable_feature_importance_threshold,
            )
        )
        training_features_per_method["univariable_feature_importance"] = (
            get_alphabetical_features(features=training_features)
        )
    else:
        univariable_feature_importance = None

    if filter_feature_importance:
        logger.info("Filtering features via model generated feature importance")
        training_features, feature_importance = select_features_via_feature_importance(
            data=data,
            target_feature=target_feature,
            model_choice=feature_importance_model_choice,
            param_grid=feature_importance_param_grid,
            training_features=training_features,
            feature_importance_cut_off=feature_importance_threshold,
            test_size=test_size,
            random_state=random_state,
        )
        training_features_per_method["feature_importance"] = get_alphabetical_features(
            features=training_features
        )
    else:
        feature_importance = None

    # Sorting the features alphabetically
    training_features = get_alphabetical_features(features=training_features)

    logger.info("Feature selection completed")
    return (
        training_features,
        training_features_per_method,
        univariable_feature_importance,
        feature_importance,
    )


def numeric_only_training_features(data, target_feature):
    return data.drop(columns=[target_feature]).select_dtypes(include="number").columns


def filter_out_zero_or_near_zero_variance_features(
    data, training_features, threshold=0.01
):
    from sklearn.feature_selection import VarianceThreshold

    variance_selector = VarianceThreshold(threshold=threshold)
    if data[training_features].var().max() < threshold:
        logger.warning(
            "No feature in X meets the variance threshold. Returning original features."
        )
        return training_features
    else:
        variance_selector.fit(data[training_features])
        training_features = data[training_features].columns[
            variance_selector.get_support()
        ]
        return training_features


def select_features_via_feature_importance(
    data,
    target_feature,
    model_choice,
    param_grid,
    training_features,
    feature_importance_cut_off,
    test_size,
    random_state,
):

    best_model, best_params, best_rmse, feature_importance = generate_model(
        data=data,
        target_feature=target_feature,
        model_choice=model_choice,
        param_grid=param_grid,
        training_features=training_features,
        generate_feature_importance=True,
        feature_importance_path=None,
        test_size=test_size,
        random_state=random_state,
    )
    feature_selected_features = apply_filter_to_feature_importance(
        feature_importance=feature_importance, cut_off=feature_importance_cut_off
    )

    return feature_selected_features, feature_importance


def apply_filter_to_feature_importance(feature_importance, cut_off):
    feature_importance = feature_importance[feature_importance["IMPORTANCE"] > cut_off]
    feature_selected_features = feature_importance["FEATURES"].tolist()
    return feature_selected_features


def select_features_via_univariable_feature_importance(
    data, target_feature, training_features, method, threshold
):
    univariable_feature_importance = generate_univariable_feature_importance(
        data=data,
        target_feature=target_feature,
        training_features=training_features,
        method=method,
    )

    # Apply the threshold
    univariable_feature_importance = univariable_feature_importance[
        abs(univariable_feature_importance["IMPORTANCE"]) > threshold
    ]
    feature_selected_features = univariable_feature_importance["FEATURES"].tolist()

    return feature_selected_features, univariable_feature_importance


def get_alphabetical_features(features):
    return np.sort(features)
