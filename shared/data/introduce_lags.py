import logging as logger
import numpy as np
from statsmodels.tsa.stattools import acf
from statsmodels.tsa.stattools import pacf
from statsmodels.tsa.stattools import ccf
from statsmodels.tsa.stattools import grangercausalitytests
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


def introduce_lags(
    data,
    timestamp,
    features,
    lags,
    optimise_lags=False,
    target=None,
    max_lag=144,
    overwrite_existing_features=True,
):
    logger.info("Starting lag introduction process.")
    logger.info("Lags to be introduced: %s", ", ".join([str(lag) for lag in lags]))

    data = sort_data_by_timestamp(data=data, timestamp=timestamp)
    logger.info(f"Data sorted by timestamp: {timestamp}")

    for feature, lag in zip(features, lags):
        if optimise_lags:
            logger.info(f"Optimising lag for feature: {feature}")
            optimal_lag, optimal_lag_identification_dict = optimal_lag_identification(
                data=data, feature=feature, target=target, max_lag=max_lag
            )
            logger.info(f"Processing feature: {feature} with lag: {optimal_lag}")
            data = introduce_lag(
                data=data,
                feature=feature,
                lag=optimal_lag,
                overwrite_existing_features=overwrite_existing_features,
            )
        else:
            logger.info(f"Processing feature: {feature} with lag: {lag}")
            data = introduce_lag(
                data=data,
                feature=feature,
                lag=lag,
                overwrite_existing_features=overwrite_existing_features,
            )

    return data


def sort_data_by_timestamp(data, timestamp):
    return data.sort_values(by=timestamp)


def introduce_lag(data, feature, lag, overwrite_existing_features):
    if overwrite_existing_features:
        data[feature] = data[feature].shift(lag)
    else:
        data[f"{feature}_lag_{lag}"] = data[feature].shift(lag)
    return data


def optimal_lag_identification(data, feature, target, max_lag):

    data = preprocess_data_for_optimal_lag_identification(
        data=data, target=target, feature=feature
    )
    optimal_lag_via_acf = optimal_lag_identification_via_acf(data, feature, max_lag)
    optimal_lag_via_pacf = optimal_lag_identification_via_pacf(data, feature, max_lag)
    optimal_lag_via_ccf = optimal_lag_identification_via_ccf(
        data, feature, target, max_lag
    )
    optimal_lag_via_granger_causality = (
        optimal_lag_identification_via_the_granger_causality_test(
            data, feature, target, max_lag
        )
    )
    optimal_lag_via_lagged_regression = (
        optimal_lag_identification_via_lagged_regression(data, feature, target, max_lag)
    )

    optimal_lag_dict = {
        "acf": optimal_lag_via_acf,
        "pacf": optimal_lag_via_pacf,
        "ccf": optimal_lag_via_ccf,
        "granger_causality": optimal_lag_via_granger_causality,
        "lagged_regression": optimal_lag_via_lagged_regression,
    }

    optimal_lag = np.mean(
        [
            optimal_lag_via_acf,
            optimal_lag_via_pacf,
            optimal_lag_via_ccf,
            optimal_lag_via_granger_causality,
            optimal_lag_via_lagged_regression,
        ]
    )
    optimal_lag = int(np.round(optimal_lag))
    logger.info(f"Average optimal lag for feature {feature} is {optimal_lag}")

    return optimal_lag, optimal_lag_dict


def preprocess_data_for_optimal_lag_identification(data, target, feature):
    data = data[[target, feature]].ffill().bfill()
    return data


def optimal_lag_identification_via_acf(data, feature, max_lag):
    acf_values = acf(data[feature], nlags=max_lag)
    optimal_lag = np.argmax(acf_values[1:]) + 1  # +1 because acf_values[0] is lag 0

    logger.info(
        f"Optimal lag for feature {feature} is {optimal_lag} with ACF value {acf_values[optimal_lag]}"
    )
    return optimal_lag


def optimal_lag_identification_via_pacf(data, feature, max_lag):
    pacf_values = pacf(data[feature], nlags=max_lag)
    optimal_lag = np.argmax(pacf_values[1:]) + 1  # +1 because pacf_values[0] is lag 0

    logger.info(
        f"Optimal lag for feature {feature} is {optimal_lag} with PACF value {pacf_values[optimal_lag]}"
    )
    return optimal_lag


def optimal_lag_identification_via_ccf(data, feature, target, max_lag):
    ccf_values = ccf(data[feature], data[target])[: max_lag + 1]
    optimal_lag = np.argmax(
        np.abs(ccf_values)
    )  # Use absolute values to find the strongest correlation

    logger.info(
        f"Optimal lag between feature {feature} and target {target} is {optimal_lag} with CCF value {ccf_values[optimal_lag]}"
    )
    return optimal_lag


def optimal_lag_identification_via_the_granger_causality_test(
    data, feature, target, max_lag
):
    test_result = grangercausalitytests(data, max_lag, verbose=False)
    p_values = [round(test_result[i + 1][0]["ssr_ftest"][1], 4) for i in range(max_lag)]
    optimal_lag = np.argmin(p_values) + 1  # +1 because lags are 1-indexed

    logger.info(
        f"Optimal lag between feature {feature} and target {target} via Granger Causality Test is {optimal_lag} with p-value {p_values[optimal_lag-1]}"
    )
    return optimal_lag


def optimal_lag_identification_via_lagged_regression(data, feature, target, max_lag):

    min_mse = float("inf")
    optimal_lag = 0

    for lag in range(1, max_lag + 1):
        lagged_feature = data[feature].shift(lag).dropna()
        aligned_target = data[target][lagged_feature.index]

        model = LinearRegression()
        model.fit(lagged_feature.values.reshape(-1, 1), aligned_target)
        predictions = model.predict(lagged_feature.values.reshape(-1, 1))
        mse = mean_squared_error(aligned_target, predictions)

        if mse < min_mse:
            min_mse = mse
            optimal_lag = lag

    logger.info(
        f"Optimal lag between feature {feature} and target {target} via Lagged Regression is {optimal_lag} with MSE {min_mse}"
    )
    return optimal_lag
