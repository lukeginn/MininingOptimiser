import pandas as pd
import logging as logger
from dataclasses import dataclass, field
from typing import List, Any

@dataclass
class MissingDataIdentifier:
    data: pd.DataFrame
    timestamp: str
    features: List[str] = field(default_factory=list)
    unique_values_identification: bool = True
    unique_values_threshold: int = 5
    explicit_missing_values: bool = True
    explicit_missing_indicators: List[Any] = field(default_factory=lambda: [0, pd.NA, None, -999, -9999, "?"])
    repeating_values: bool = True
    repeating_values_threshold: int = 5
    repeating_values_proportion_threshold: float = 0.8

    def run(self):
        logger.info("Starting missing data identification process.")
        logger.info(
            "Procedures to be run: %s",
            ", ".join(
                [
                    "Identifying unique values" if self.unique_values_identification else "",
                    (
                        "Identifying explicit missing indicators"
                        if self.explicit_missing_values
                        else ""
                    ),
                    "Identifying repeating values" if self.repeating_values else "",
                ]
            ).strip(", "),
        )

        if not self.features:
            logger.info("Identifying numerical features.")
            self.features = self._numerical_features()

        self.data = self._sort_data_by_timestamp()
        logger.info(f"Data sorted by timestamp: {self.timestamp}")

        for feature in self.features:
            logger.info(f"Processing feature: {feature}")

            if self.unique_values_identification:
                if self._identify_if_the_feature_has_low_unique_values(feature):
                    logger.info(f"Feature '{feature}' skipped due to low unique values.")
                    continue

            if self.explicit_missing_values:
                self.data = self._identify_explicit_missing_indicators(feature)

            if self.repeating_values:
                self.data = self._identify_repeat_values(feature)

        logger.info("Completed missing data identification process.")
        return self.data

    def _numerical_features(self):
        return self.data.select_dtypes(include="number").columns.tolist()

    def _sort_data_by_timestamp(self):
        return self.data.sort_values(by=self.timestamp)

    def _identify_if_the_feature_has_low_unique_values(self, feature):
        unique_values_count = self.data[feature].nunique()
        if unique_values_count <= self.unique_values_threshold:
            return True
        return False

    def _identify_explicit_missing_indicators(self, feature):
        self.data[feature] = self.data[feature].replace(self.explicit_missing_indicators, pd.NA)
        return self.data

    def _identify_repeat_values(self, feature):
        # Identify segments where the feature value repeats
        self.data["repeat_count"] = (
            self.data[feature]
            .groupby((self.data[feature] != self.data[feature].shift()).cumsum())
            .transform("size")
        )

        # Calculate the proportion of repeating values in the feature
        repeating_proportion = (self.data["repeat_count"] > self.repeating_values_threshold).mean()

        # If the proportion of repeating values is high, skip this feature
        if repeating_proportion > self.repeating_values_proportion_threshold:
            logger.info(
                f"Feature '{feature}' skipped due to high proportion of repeating values."
            )
            self.data.drop(columns=["repeat_count"], inplace=True)
            return self.data

        # Turn the entire segment into missing values if it repeats more than the repeating_values_threshold
        if (self.data["repeat_count"] > self.repeating_values_threshold).any():
            logger.info(
                f"Feature '{feature}' has segments with repeating values exceeding the threshold."
            )
        self.data.loc[self.data["repeat_count"] > self.repeating_values_threshold, feature] = pd.NA

        # Drop the temporary 'repeat_count' column
        self.data.drop(columns=["repeat_count"], inplace=True)

        return self.data
