import pandas as pd
import logging as logger


def identify_missing_data(
    data,
    timestamp,
    features=None,
    unique_values_identification=True,
    unique_values_threshold=5,
    explicit_missing_values=True,
    explicit_missing_indicators=[0, pd.NA, None, -999, -9999, "?"],
    repeating_values=True,
    repeating_values_threshold=5,
    repeating_values_proportion_threshold=0.8,
):
    logger.info("Starting missing data identification process.")
    logger.info(
        "Procedures to be run: %s",
        ", ".join(
            [
                "Identifying unique values" if unique_values_identification else "",
                (
                    "Identifying explicit missing indicators"
                    if explicit_missing_values
                    else ""
                ),
                "Identifying repeating values" if repeating_values else "",
            ]
        ).strip(", "),
    )

    if features is None:
        logger.info("Identifying numerical features.")
        features = numerical_features(data=data)

    data = sort_data_by_timestamp(data=data, timestamp=timestamp)
    logger.info(f"Data sorted by timestamp: {timestamp}")

    for feature in features:
        logger.info(f"Processing feature: {feature}")

        if unique_values_identification:
            if identify_if_the_feature_has_low_unique_values(
                data=data,
                feature=feature,
                unique_values_threshold=unique_values_threshold,
            ):
                logger.info(f"Feature '{feature}' skipped due to low unique values.")
                continue

        if explicit_missing_values:
            data = identify_explicit_missing_indicators(
                data=data,
                feature=feature,
                missing_indicators=explicit_missing_indicators,
            )

        if repeating_values:
            data = identify_repeat_values(
                data=data,
                feature=feature,
                repeating_values_threshold=repeating_values_threshold,
                repeating_values_proportion_threshold=repeating_values_proportion_threshold,
            )

    logger.info("Completed missing data identification process.")
    return data


def numerical_features(data):
    return data.select_dtypes(include="number").columns


def sort_data_by_timestamp(data, timestamp):
    return data.sort_values(by=timestamp)


def identify_if_the_feature_has_low_unique_values(
    data, feature, unique_values_threshold
):
    unique_values_count = data[feature].nunique()
    if unique_values_count <= unique_values_threshold:
        return True
    return False


def identify_explicit_missing_indicators(data, feature, missing_indicators):
    data[feature] = data[feature].replace(missing_indicators, pd.NA)
    return data


def identify_repeat_values(
    data, feature, repeating_values_threshold, repeating_values_proportion_threshold=0.8
):
    # Identify segments where the feature value repeats
    data["repeat_count"] = (
        data[feature]
        .groupby((data[feature] != data[feature].shift()).cumsum())
        .transform("size")
    )

    # Calculate the proportion of repeating values in the feature
    repeating_proportion = (data["repeat_count"] > repeating_values_threshold).mean()

    # If the proportion of repeating values is high, skip this feature
    if (
        repeating_proportion > repeating_values_proportion_threshold
    ):  # You can adjust this threshold as needed
        logger.info(
            f"Feature '{feature}' skipped due to high proportion of repeating values."
        )
        data.drop(columns=["repeat_count"], inplace=True)
        return data

    # Turn the entire segment into missing values if it repeats more than the repeating_values_threshold
    if (data["repeat_count"] > repeating_values_threshold).any():
        logger.info(
            f"Feature '{feature}' has segments with repeating values exceeding the threshold."
        )
    data.loc[data["repeat_count"] > repeating_values_threshold, feature] = pd.NA

    # Drop the temporary 'repeat_count' column
    data.drop(columns=["repeat_count"], inplace=True)

    return data
