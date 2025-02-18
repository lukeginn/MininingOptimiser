import pandas as pd
import logging as logger


def aggregate_data_via_timestamp(
    data,
    timestamp,
    features_to_aggregate=None,
    aggregation_types=["mean", "std", "min", "max"],
):

    data = format_timestamp(data=data, timestamp=timestamp)

    if features_to_aggregate is None:
        features_to_aggregate = identify_features_to_aggregate(data=data)

    data = aggregate_using_timestamp(
        data=data,
        timestamp=timestamp,
        features_to_aggregate=features_to_aggregate,
        aggregation_types=aggregation_types,
    )

    return data


def rolling_aggregate_data_via_timestamp(
    data,
    timestamp,
    features_to_aggregate=None,
    window=144,
    min_periods=144,
    window_selection_frequency=144,
    aggregation_types=["mean", "std", "min", "max"],
):
    data = sort_data_by_timestamp(data=data, timestamp=timestamp)

    if features_to_aggregate is None:
        features_to_aggregate = identify_features_to_aggregate(data=data)

    data = rolling_aggregate_using_timestamp(
        data=data,
        timestamp=timestamp,
        features_to_aggregate=features_to_aggregate,
        aggregation_types=aggregation_types,
        window=window,
        min_periods=min_periods,
    )

    data = select_every_nth_row(data=data, n=window_selection_frequency)

    return data


def identify_features_to_aggregate(data):
    # Identify the features to aggregate
    features_to_aggregate = data.select_dtypes(include="number").columns.tolist()

    return features_to_aggregate


def aggregate_using_timestamp(
    data, timestamp, features_to_aggregate, aggregation_types
):

    if not isinstance(aggregation_types, list):
        aggregation_types = [aggregation_types]

    # Aggregate the data
    data = data.groupby(timestamp).agg(
        {feature: aggregation_types for feature in features_to_aggregate}
    )

    # Flatten the column names
    data.columns = ["_".join(x) for x in data.columns.to_flat_index()]

    return data


def format_timestamp(data, timestamp, freq="D"):

    # Format the timestamp into the specified frequency
    data[timestamp] = data[timestamp].dt.floor(freq)

    return data


def rolling_aggregate_using_timestamp(
    data, timestamp, features_to_aggregate, aggregation_types, window, min_periods
):

    if not isinstance(aggregation_types, list):
        aggregation_types = [aggregation_types]

    # Apply rolling aggregation
    rolling_data = (
        data[features_to_aggregate]
        .rolling(window=window, min_periods=min_periods)
        .agg(aggregation_types)
    )

    # Flatten the column names
    rolling_data.columns = ["_".join(x) for x in rolling_data.columns.to_flat_index()]

    # Combine the rolling data with the original timestamp
    data = pd.concat([data[timestamp], rolling_data], axis=1).reset_index(drop=True)

    # Filter out the first few rows with NaN values from min_periods
    data = data.iloc[min_periods:]

    return data


def sort_data_by_timestamp(data, timestamp):

    data = data.sort_values(by=timestamp).reset_index(drop=True)

    return data


def select_every_nth_row(data, n):

    # Ensure n is an integer and at least 1
    n = max(1, int(round(n)))

    data = data.iloc[::n]

    return data
