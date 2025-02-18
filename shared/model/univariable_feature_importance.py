import logging as logger


def generate_univariable_feature_importance(
    data, target_feature, path=None, training_features=None, method="kendall"
):

    logger.info("Running univariable feature importance")

    if training_features is None:
        logger.info("No training features provided. Using all numeric features")
        training_features = numeric_only_training_features(
            data=data, target_feature=target_feature
        )

    univariable_feature_importance = run_univariable_feature_importance_analysis(
        data=data,
        target_feature=target_feature,
        training_features=training_features,
        method=method,
    )

    if path is not None:
        write_univariable_corr_to_file(
            univariable_feature_importance=univariable_feature_importance, path=path
        )

    return univariable_feature_importance


def numeric_only_training_features(data, target_feature):
    return data.drop(columns=[target_feature]).select_dtypes(include="number").columns


def run_univariable_feature_importance_analysis(
    data, target_feature, training_features, method
):
    if target_feature not in training_features:
        training_features = list(training_features) + [target_feature]
    data = data[training_features]

    if method == "pearson":
        # This method is linear and assumes normal distribution
        # This method is monotonic (due to linearity)
        # This method is sensitive to outliers
        univariable_feature_importance = (
            data.corr(method="pearson")[target_feature]
            .abs()
            .sort_values(ascending=False)
        )
        univariable_feature_importance = data.corr(method="pearson")[
            target_feature
        ].reindex(univariable_feature_importance.index)
    elif method == "spearman":
        # This method is non-linear and does not assume normal distribution
        # This method is monotonic
        # This method is less sensitive to outliers
        univariable_feature_importance = (
            data.corr(method="spearman")[target_feature]
            .abs()
            .sort_values(ascending=False)
        )
        univariable_feature_importance = data.corr(method="spearman")[
            target_feature
        ].reindex(univariable_feature_importance.index)
    elif method == "kendall":
        # This method is non-linear and does not assume a normal distribution
        # This method is monotonic
        # This method is less sensitive to outliers
        # Generally it's better than spearman for all sample sizes, but it's much more computationally expensive
        univariable_feature_importance = (
            data.corr(method="kendall")[target_feature]
            .abs()
            .sort_values(ascending=False)
        )
        univariable_feature_importance = data.corr(method="kendall")[
            target_feature
        ].reindex(univariable_feature_importance.index)
    else:
        raise ValueError("Unsupported method. Use 'pearson', 'spearman', or 'kendall'.")

    # Convert the univariable_feature_importance to a DataFrame
    univariable_feature_importance = univariable_feature_importance.reset_index()
    univariable_feature_importance.columns = ["FEATURES", "IMPORTANCE"]

    # Remove the target from the univariable_feature_importance
    univariable_feature_importance = univariable_feature_importance[
        univariable_feature_importance["FEATURES"] != target_feature
    ]

    return univariable_feature_importance


def write_univariable_corr_to_file(univariable_feature_importance, path):
    univariable_feature_importance.to_csv(path, index=False)
    logger.info(f"Univariable correlation written to {path}")
