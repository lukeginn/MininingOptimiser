import logging as logger
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class UnivariableFeatureImportanceProcessor:
    data: pd.DataFrame
    target_feature: str
    path: Optional[str] = None
    training_features: Optional[List[str]] = field(default_factory=list)
    method: str = "kendall"

    def run(self) -> pd.DataFrame:
        logger.info("Running univariable feature importance")

        if not self.training_features:
            logger.info("No training features provided. Using all numeric features")
            self.training_features = self._numeric_only_training_features()

        univariable_feature_importance = self._univariable_feature_importance_analysis()

        if self.path is not None:
            self._write_univariable_corr_to_file(univariable_feature_importance)

        return univariable_feature_importance

    def _numeric_only_training_features(self) -> List[str]:
        return (
            self.data.drop(columns=[self.target_feature])
            .select_dtypes(include="number")
            .columns.tolist()
        )

    def _univariable_feature_importance_analysis(self) -> pd.DataFrame:
        if self.target_feature not in self.training_features:
            self.training_features.append(self.target_feature)
        data = self.data[self.training_features]

        if self.method == "pearson":
            univariable_feature_importance = (
                data.corr(method="pearson")[self.target_feature]
                .abs()
                .sort_values(ascending=False)
            )
            univariable_feature_importance = data.corr(method="pearson")[
                self.target_feature
            ].reindex(univariable_feature_importance.index)
        elif self.method == "spearman":
            univariable_feature_importance = (
                data.corr(method="spearman")[self.target_feature]
                .abs()
                .sort_values(ascending=False)
            )
            univariable_feature_importance = data.corr(method="spearman")[
                self.target_feature
            ].reindex(univariable_feature_importance.index)
        elif self.method == "kendall":
            univariable_feature_importance = (
                data.corr(method="kendall")[self.target_feature]
                .abs()
                .sort_values(ascending=False)
            )
            univariable_feature_importance = data.corr(method="kendall")[
                self.target_feature
            ].reindex(univariable_feature_importance.index)
        else:
            raise ValueError(
                "Unsupported method. Use 'pearson', 'spearman', or 'kendall'."
            )

        univariable_feature_importance = univariable_feature_importance.reset_index()
        univariable_feature_importance.columns = ["FEATURES", "IMPORTANCE"]
        univariable_feature_importance = univariable_feature_importance[
            univariable_feature_importance["FEATURES"] != self.target_feature
        ]

        return univariable_feature_importance

    def _write_univariable_corr_to_file(
        self, univariable_feature_importance: pd.DataFrame
    ) -> None:
        univariable_feature_importance.to_csv(self.path, index=False)
        logger.info(f"Univariable correlation written to {self.path}")
