import logging as logger
import numpy as np


def missing_data_correction(
    data,
    delete_all_rows_with_missing_values=False,
    interpolate_time_series=False,
    interpolate_highly_regular_time_series=False,
    replace_missing_values_with_x=True,
    replace_missing_values_with_last_known_value=False,
    interpolate_method="linear",
    interpolate_limit_direction="both",
    interpolate_max_gap=5,
    interpolate_highly_regular_method="linear",
    interpolate_highly_regular_limit_direction="both",
    interpolate_highly_regular_interval_min=144,
    replace_missing_values_with_last_known_value_backfill=False,
    timestamp=None,
    features=None,
    x=0,
):

    logger.info("Correcting missing data")
    if features is None:
        features = numerical_features(data=data)
        features = features_with_variance(data=data)

    if interpolate_time_series:
        data = interpolate_time_series_features(
            data=data,
            timestamp=timestamp,
            features=features,
            method=interpolate_method,
            limit_direction=interpolate_limit_direction,
            max_gap=interpolate_max_gap,
        )
    if interpolate_highly_regular_time_series:
        data = interpolate_highly_regular_time_series_features(
            data=data,
            timestamp=timestamp,
            features=features,
            method=interpolate_highly_regular_method,
            limit_direction=interpolate_highly_regular_limit_direction,
            regular_interval_min=interpolate_highly_regular_interval_min,
        )
    if replace_missing_values_with_x or replace_missing_values_with_last_known_value:
        data = replace_missing_values(
            data=data,
            features=features,
            value_to_replace=x,
            last_known_value=replace_missing_values_with_last_known_value,
            backfill=replace_missing_values_with_last_known_value_backfill,
        )

    if delete_all_rows_with_missing_values:
        data = delete_rows_with_missing_values(data=data, features=features)

    return data


def numerical_features(data):
    return data.select_dtypes(include="number").columns


def features_with_variance(data):
    # Select features that have at least one non-missing value
    features = data.columns[data.notna().any()].tolist()
    return features


def delete_rows_with_missing_values(data, features):
    logger.info("Deleting rows with missing values")
    data = data.dropna(subset=features)

    return data


def replace_missing_values(
    data, features, value_to_replace, last_known_value, backfill
):
    if last_known_value:
        logger.info("Replacing missing values with last known value per feature")
        for feature in features:
            data[feature] = data[feature].fillna(method="ffill")
            if backfill:
                data[feature] = data[feature].fillna(method="bfill")
        return data
    else:
        logger.info(f"Replacing missing values with {value_to_replace}")
        for feature in features:
            data[feature] = data[feature].fillna(value_to_replace)

    return data


def interpolate_time_series_features(
    data, timestamp, features, method, limit_direction, max_gap
):
    logger.info("Interpolating time series features")
    if features is None:
        features = numerical_features(data=data)
    data = sort_data_by_timestamp(data=data, timestamp=timestamp)
    data = interpolate_features(
        data=data,
        features=features,
        method=method,
        limit_direction=limit_direction,
        max_gap=max_gap,
    )

    return data


def interpolate_highly_regular_time_series_features(
    data, timestamp, features, method, limit_direction, regular_interval_min
):
    logger.info("Interpolating time series features")
    if features is None:
        features = numerical_features(data=data)
    data = sort_data_by_timestamp(data=data, timestamp=timestamp)
    data = interpolate_highly_regular_features(
        data=data,
        features=features,
        method=method,
        limit_direction=limit_direction,
        regular_interval_min=regular_interval_min,
    )

    return data


def numerical_features(data):
    return data.select_dtypes(include="number").columns


def sort_data_by_timestamp(data, timestamp):
    data = data.sort_values(by=timestamp)
    logger.info("Data sorted by timestamp")

    return data


def interpolate_features(data, features, method, limit_direction, max_gap):
    for feature in features:
        data = interpolate_one_feature(
            data=data,
            feature=feature,
            method=method,
            limit_direction=limit_direction,
            max_gap=max_gap,
        )
    logger.info("Features interpolated")

    return data


def interpolate_highly_regular_features(
    data, features, method, limit_direction, regular_interval_min
):
    for feature in features:
        data = interpolate_one_feature_with_high_and_regular_sparsity(
            data=data,
            feature=feature,
            method=method,
            limit_direction=limit_direction,
            regular_interval_min=regular_interval_min,
        )
    logger.info("Features interpolated")

    return data


def interpolate_one_feature(data, feature, method, limit_direction, max_gap):
    # Check if the feature is a datetime feature, if it is, skip it
    if np.issubdtype(data[feature].dtype, np.datetime64):
        logger.info(f"Skipping datetime feature: {feature}")
        return data

    # Identify the gaps in the data
    is_nan = data[feature].isna()
    gap_sizes = is_nan.groupby((is_nan != is_nan.shift()).cumsum()).transform("sum")

    # Make a copy of the original feature
    original_feature = data[feature].copy()


    # Interpolate the missing values
    if method == "bfill":
        data[feature] = data[feature].fillna(method="bfill")
    else:
        data[feature] = data[feature].interpolate(
            method=method, limit_direction=limit_direction
        )

    # Where the gap_sizes are greater than the max_gap, replace the values with the original values
    data[feature] = data[feature].where(gap_sizes <= max_gap, other=original_feature)

    return data


def interpolate_one_feature_with_high_and_regular_sparsity(
    data, feature, method, limit_direction, regular_interval_min
):
    # Check if the feature has regular intervals of non-missing values
    non_missing_indices = data[feature].dropna().index
    if len(non_missing_indices) > 1:
        intervals = non_missing_indices.to_series().diff().dropna()
        if (intervals >= regular_interval_min).mode().all():
            logger.info(
                f"Feature {feature} has regular intervals of non-missing values"
            )

            # Interpolate the missing values
            if method == "bfill":
                data[feature] = data[feature].fillna(method="bfill")
            else:
                data[feature] = data[feature].interpolate(
                    method=method, limit_direction=limit_direction
                )
        else:
            logger.info(
                f"Feature {feature} does not have regular intervals of non-missing values"
            )
    else:
        logger.info(
            f"Feature {feature} does not have enough non-missing values to determine regular intervals"
        )

    return data
