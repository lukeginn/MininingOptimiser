import numpy as np
import logging as logger
from shared.model.classes.model_processor import ModelProcessor
from shared.model.classes.univariable_feature_importance_processor import (
    UnivariableFeatureImportanceProcessor,
)
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
import pandas as pd


@dataclass
class FeatureSelectorProcessor:
    data: pd.DataFrame
    target_feature: str
    filter_low_variance: bool = True
    filter_univariable_feature_importance: bool = True
    filter_feature_importance: bool = False
    low_variance_threshold: float = 0.01
    univariable_feature_importance_method: str = "kendall"
    univariable_feature_importance_threshold: float = 1
    feature_importance_model_choice: Optional[str] = None
    feature_importance_param_grid: Optional[Dict[str, Any]] = None
    feature_importance_threshold: float = 1
    training_features: Optional[List[str]] = None
    test_size: float = 0.2
    random_state: int = 42

    def run(self) -> Dict[str, Any]:
        if self.training_features is None:
            logger.info("No training features provided. Using all numeric features")
            self.training_features = self._numeric_only_training_features()

        # Dictionary to store the features selected by each method
        training_features_per_method = dict()

        if self.filter_low_variance:
            logger.info("Filtering out zero or near zero variance features")
            self.training_features = (
                self._filter_out_zero_or_near_zero_variance_features()
            )
            training_features_per_method["low_variance"] = (
                self._get_alphabetical_features()
            )

        if self.filter_univariable_feature_importance:
            logger.info("Filtering features via univariable feature importance")
            self.training_features, univariable_feature_importance = (
                self._select_features_via_univariable_feature_importance()
            )
            training_features_per_method["univariable_feature_importance"] = (
                self._get_alphabetical_features()
            )
        else:
            univariable_feature_importance = None

        if self.filter_feature_importance:
            logger.info("Filtering features via model generated feature importance")
            self.training_features, feature_importance = (
                self._select_features_via_feature_importance()
            )
            training_features_per_method["feature_importance"] = (
                self._get_alphabetical_features()
            )
        else:
            feature_importance = None

        # Sorting the features alphabetically
        self.training_features = self._get_alphabetical_features()

        logger.info("Feature selection completed")
        return {
            "training_features": self.training_features,
            "training_features_per_method": training_features_per_method,
            "univariable_feature_importance": univariable_feature_importance,
            "feature_importance": feature_importance,
        }

    def _numeric_only_training_features(self) -> List[str]:
        return (
            self.data.drop(columns=[self.target_feature])
            .select_dtypes(include="number")
            .columns.tolist()
        )

    def _filter_out_zero_or_near_zero_variance_features(self) -> List[str]:
        from sklearn.feature_selection import VarianceThreshold

        variance_selector = VarianceThreshold(threshold=self.low_variance_threshold)
        if self.data[self.training_features].var().max() < self.low_variance_threshold:
            logger.warning(
                "No feature in X meets the variance threshold. Returning original features."
            )
            return self.training_features
        else:
            variance_selector.fit(self.data[self.training_features])
            return (
                self.data[self.training_features]
                .columns[variance_selector.get_support()]
                .tolist()
            )

    def _select_features_via_feature_importance(self) -> (List[str], pd.DataFrame):
        model_processor = ModelProcessor(
            data=self.data,
            target_feature=self.target_feature,
            model_choice=self.feature_importance_model_choice,
            param_grid=self.feature_importance_param_grid,
            training_features=self.training_features,
            test_size=self.test_size,
            random_state=self.random_state,
        )
        best_model, best_params, best_rmse, feature_importance = model_processor.run()
        feature_selected_features = self._apply_filter_to_feature_importance(
            feature_importance
        )
        return feature_selected_features, feature_importance

    def _apply_filter_to_feature_importance(
        self, feature_importance: pd.DataFrame
    ) -> List[str]:
        feature_importance = feature_importance[
            feature_importance["IMPORTANCE"] > self.feature_importance_threshold
        ]
        return feature_importance["FEATURES"].tolist()

    def _select_features_via_univariable_feature_importance(
        self,
    ) -> (List[str], pd.DataFrame):
        univariable_feature_importance_processor = (
            UnivariableFeatureImportanceProcessor(
                data=self.data,
                target_feature=self.target_feature,
                training_features=self.training_features,
                method=self.univariable_feature_importance_method,
            )
        )
        univariable_feature_importance = univariable_feature_importance_processor.run()

        # Apply the threshold
        univariable_feature_importance = univariable_feature_importance[
            abs(univariable_feature_importance["IMPORTANCE"])
            > self.univariable_feature_importance_threshold
        ]
        feature_selected_features = univariable_feature_importance["FEATURES"].tolist()

        return feature_selected_features, univariable_feature_importance

    def _get_alphabetical_features(self) -> List[str]:
        return np.sort(self.training_features).tolist()
