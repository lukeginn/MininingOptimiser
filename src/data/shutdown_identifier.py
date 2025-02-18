import logging as logger


def identify_shutdowns(
    data,
    shutdown_features,
    cutoff_values
):
    for feature in shutdown_features:
        data[feature] = data[feature].mask(data[feature] < cutoff_values[feature])
    return data