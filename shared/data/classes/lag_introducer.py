import logging as logger
import numpy as np
from statsmodels.tsa.stattools import acf, pacf, ccf, grangercausalitytests
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from dataclasses import dataclass, field
from typing import List, Optional
import pandas as pd

@dataclass
class LagIntroducer:
    data: pd.DataFrame
    timestamp: str
    features: List[str]
    lags: List[int]
    optimise_lags: bool = False
    target: Optional[str] = None
    max_lag: int = 144
    overwrite_existing_features: bool = True

    def run(self) -> pd.DataFrame:
        logger.info("Starting lag introduction process.")
        logger.info("Lags to be introduced: %s", ", ".join([str(lag) for lag in self.lags]))

        self.data = self._sort_data_by_timestamp()
        logger.info(f"Data sorted by timestamp: {self.timestamp}")

        for feature, lag in zip(self.features, self.lags):
            if self.optimise_lags:
                logger.info(f"Optimising lag for feature: {feature}")
                optimal_lag, optimal_lag_identification_dict = self._optimal_lag_identification(feature)
                logger.info(f"Processing feature: {feature} with lag: {optimal_lag}")
                self.data = self._introduce_lag(feature, optimal_lag)
            else:
                logger.info(f"Processing feature: {feature} with lag: {lag}")
                self.data = self._introduce_lag(feature, lag)

        return self.data

    def _sort_data_by_timestamp(self) -> pd.DataFrame:
        return self.data.sort_values(by=self.timestamp)

    def _introduce_lag(self, feature: str, lag: int) -> pd.DataFrame:
        if self.overwrite_existing_features:
            self.data[feature] = self.data[feature].shift(lag)
        else:
            self.data[f"{feature}_lag_{lag}"] = self.data[feature].shift(lag)
        return self.data

    def _optimal_lag_identification(self, feature: str) -> (int, dict):
        data = self._preprocess_data_for_optimal_lag_identification(feature)
        optimal_lag_via_acf = self._optimal_lag_identification_via_acf(feature)
        optimal_lag_via_pacf = self._optimal_lag_identification_via_pacf(feature)
        optimal_lag_via_ccf = self._optimal_lag_identification_via_ccf(feature)
        optimal_lag_via_granger_causality = self._optimal_lag_identification_via_granger_causality(feature)
        optimal_lag_via_lagged_regression = self._optimal_lag_identification_via_lagged_regression(feature)

        optimal_lag_dict = {
            "acf": optimal_lag_via_acf,
            "pacf": optimal_lag_via_pacf,
            "ccf": optimal_lag_via_ccf,
            "granger_causality": optimal_lag_via_granger_causality,
            "lagged_regression": optimal_lag_via_lagged_regression,
        }

        optimal_lag = np.mean([
            optimal_lag_via_acf,
            optimal_lag_via_pacf,
            optimal_lag_via_ccf,
            optimal_lag_via_granger_causality,
            optimal_lag_via_lagged_regression,
        ])
        optimal_lag = int(np.round(optimal_lag))
        logger.info(f"Average optimal lag for feature {feature} is {optimal_lag}")

        return optimal_lag, optimal_lag_dict

    def _preprocess_data_for_optimal_lag_identification(self, feature: str) -> pd.DataFrame:
        return self.data[[self.target, feature]].ffill().bfill()

    def _optimal_lag_identification_via_acf(self, feature: str) -> int:
        acf_values = acf(self.data[feature], nlags=self.max_lag)
        optimal_lag = np.argmax(acf_values[1:]) + 1  # +1 because acf_values[0] is lag 0
        logger.info(f"Optimal lag for feature {feature} is {optimal_lag} with ACF value {acf_values[optimal_lag]}")
        return optimal_lag

    def _optimal_lag_identification_via_pacf(self, feature: str) -> int:
        pacf_values = pacf(self.data[feature], nlags=self.max_lag)
        optimal_lag = np.argmax(pacf_values[1:]) + 1  # +1 because pacf_values[0] is lag 0
        logger.info(f"Optimal lag for feature {feature} is {optimal_lag} with PACF value {pacf_values[optimal_lag]}")
        return optimal_lag

    def _optimal_lag_identification_via_ccf(self, feature: str) -> int:
        ccf_values = ccf(self.data[feature], self.data[self.target])[:self.max_lag + 1]
        optimal_lag = np.argmax(np.abs(ccf_values))  # Use absolute values to find the strongest correlation
        logger.info(f"Optimal lag between feature {feature} and target {self.target} is {optimal_lag} with CCF value {ccf_values[optimal_lag]}")
        return optimal_lag

    def _optimal_lag_identification_via_granger_causality(self, feature: str) -> int:
        test_result = grangercausalitytests(self.data[[self.target, feature]], self.max_lag, verbose=False)
        p_values = [round(test_result[i + 1][0]["ssr_ftest"][1], 4) for i in range(self.max_lag)]
        optimal_lag = np.argmin(p_values) + 1  # +1 because lags are 1-indexed
        logger.info(f"Optimal lag between feature {feature} and target {self.target} via Granger Causality Test is {optimal_lag} with p-value {p_values[optimal_lag-1]}")
        return optimal_lag

    def _optimal_lag_identification_via_lagged_regression(self, feature: str) -> int:
        min_mse = float("inf")
        optimal_lag = 0

        for lag in range(1, self.max_lag + 1):
            lagged_feature = self.data[feature].shift(lag).dropna()
            aligned_target = self.data[self.target][lagged_feature.index]

            model = LinearRegression()
            model.fit(lagged_feature.values.reshape(-1, 1), aligned_target)
            predictions = model.predict(lagged_feature.values.reshape(-1, 1))
            mse = mean_squared_error(aligned_target, predictions)

            if mse < min_mse:
                min_mse = mse
                optimal_lag = lag

        logger.info(f"Optimal lag between feature {feature} and target {self.target} via Lagged Regression is {optimal_lag} with MSE {min_mse}")
        return optimal_lag
