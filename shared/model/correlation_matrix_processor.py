import logging as logger
import seaborn as sns
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Optional
import pandas as pd

@dataclass
class CorrelationMatrixProcessor:
    data: pd.DataFrame
    features: List[str]
    method: str = "kendall"
    csv_path: Optional[str] = None
    plotting_path: Optional[str] = None

    def run(self) -> pd.DataFrame:
        logger.info("Generating correlation matrix")
        data = self.data[self.features]

        correlation_matrix = self._create_correlation_matrix(data=data, method=self.method)

        if self.csv_path is not None:
            self._saving_correlation_matrix(correlation_matrix=correlation_matrix, path=self.csv_path)

        if self.plotting_path is not None:
            self._plot_correlation_matrix(correlation_matrix=correlation_matrix, path=self.plotting_path)

        return correlation_matrix

    def _create_correlation_matrix(self, data: pd.DataFrame, method: str) -> pd.DataFrame:
        if method not in ["pearson", "spearman", "kendall"]:
            raise ValueError("Method must be 'pearson', 'spearman', or 'kendall'")
        correlation_matrix = data.corr(method=method)
        return correlation_matrix

    def _saving_correlation_matrix(self, correlation_matrix: pd.DataFrame, path: str) -> None:
        logger.info(f"Saving correlation matrix to {path}")
        correlation_matrix.to_csv(path)

    def _plot_correlation_matrix(self, correlation_matrix: pd.DataFrame, path: Optional[str] = None) -> None:
        logger.info(f"Plotting correlation matrix to {path}")
        plt.figure(figsize=(10, 8))  # Adjust the figure size as needed
        sns.heatmap(
            correlation_matrix,
            annot=True,
            cmap="coolwarm",
            center=0,
            annot_kws={"size": 10},
        )
        plt.xticks(rotation=45, ha="right")  # Rotate x-axis labels for better readability
        plt.yticks(rotation=0)  # Ensure y-axis labels are horizontal
        plt.tight_layout()  # Adjust layout to make sure everything fits without overlap
        if path:
            plt.savefig(path)
        plt.close()  # Close the plot to free up memory
