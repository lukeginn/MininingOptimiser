import numpy as np
import logging as logger
from sklearn.model_selection import train_test_split, ParameterGrid
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from pygam import LinearGAM, s
from pygam.terms import TermList, SplineTerm
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingRegressor
from shared.model.classes.inference_processor import InferenceProcessor
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Tuple


@dataclass
class ModelProcessor:
    data: pd.DataFrame
    target_feature: str
    model_choice: str
    param_grid: Dict[str, Any]
    metric: str = "rmse"
    training_features: Optional[List[str]] = None
    generate_feature_importance: bool = True
    feature_importance_path: Optional[str] = None
    evaluation_results_path: Optional[str] = None
    test_size: float = 0.2
    random_state: int = 42
    n_models: int = 5
    path: Optional[str] = None

    def run(self) -> Tuple[List[Any], Dict[str, Any], float, Optional[pd.DataFrame]]:
        if self.training_features is None:
            logger.info("No training features provided. Using all numeric features")
            self.training_features = self._numeric_only_training_features()

        logger.info("Splitting data into target and features")
        data, target = self._split_data_into_target_and_training_features()

        logger.info("Splitting data into training and test sets")
        X_train, X_test, y_train, y_test = self._split_data(data, target)

        logger.info("Generating models")
        np.random.seed(self.random_state)
        models, best_params, best_rmse = self._find_best_hyperparameters(
            X_train, X_test, y_train, y_test
        )

        logger.info("Models generated successfully")

        if self.generate_feature_importance:
            logger.info("Generating feature importance")
            feature_importance = self._generate_feature_importance_dataframe(
                models[0], X_train, y_train
            )
        else:
            feature_importance = None

        return models, best_params, best_rmse, feature_importance

    def _numeric_only_training_features(self) -> List[str]:
        return (
            self.data.drop(columns=[self.target_feature])
            .select_dtypes(include="number")
            .columns.tolist()
        )

    def _split_data_into_target_and_training_features(
        self,
    ) -> Tuple[pd.DataFrame, pd.Series]:
        return self.data[self.training_features], self.data[self.target_feature]

    def _split_data(
        self, X: pd.DataFrame, y: pd.Series
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state
        )
        logger.info("Data split successfully")
        return X_train, X_test, y_train, y_test

    def _find_best_hyperparameters(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series,
    ) -> Tuple[List[Any], Dict[str, Any], float]:
        best_metric_score = float("inf")
        best_params = None
        best_models = []
        evaluation_results = []

        for params in ParameterGrid(self.param_grid):
            logger.info(f"Training {self.model_choice} model with parameters: {params}")
            training_function = self._get_model_training_function()
            models = [
                training_function(X_train, y_train, params)
                for _ in range(self.n_models)
            ]

            logger.info(f"Evaluating {self.model_choice} models")
            metrics = self._evaluate_model(models, X_test, y_test)
            metric_score = metrics[self.metric]

            evaluation_results.append({**params, **metrics})

            if metric_score < best_metric_score:
                best_metric_score = metric_score
                best_params = params
                best_models = models

        self._export_evaluation_results(pd.DataFrame(evaluation_results))

        logger.info("Best hyperparameters found")
        logger.info(f"Best Parameters: {best_params}")
        logger.info(f"Best {self.metric}: {best_metric_score}")

        return best_models, best_params, best_metric_score

    def _export_evaluation_results(self, evaluation_results: pd.DataFrame) -> None:
        if self.evaluation_results_path:
            evaluation_results.to_csv(self.evaluation_results_path, index=False)
            logger.info(
                f"Evaluation results exported to {self.evaluation_results_path}"
            )

    def _get_model_training_function(self):
        if self.model_choice == "linear_regression":
            return self._train_linear_regression
        elif self.model_choice == "gam":
            return self._train_gam
        elif self.model_choice == "gbm":
            return self._train_gbm
        else:
            raise ValueError(f"Model choice {self.model_choice} not supported")

    def _train_linear_regression(
        self, data: pd.DataFrame, target: pd.Series, params: Dict[str, Any]
    ) -> LinearRegression:
        model = LinearRegression(**params)
        model.fit(data, target)
        return model

    def _train_gam(
        self, data: pd.DataFrame, target: pd.Series, params: Dict[str, Any]
    ) -> LinearGAM:
        term_list = TermList()
        for i in range(data.shape[1]):
            constraint = params["constraints"][i]
            degree = int(params["degrees"][i])
            term = SplineTerm(i, constraints=constraint, n_splines=degree)
            term_list += term
        model = LinearGAM(term_list)
        model.fit(data, target)
        return model

    def _train_gbm(
        self, data: pd.DataFrame, target: pd.Series, params: Dict[str, Any]
    ) -> GradientBoostingRegressor:
        if "monotonic_cst" in params:
            monotonic_cst = params["monotonic_cst"]
            params_without_monotonic_cst = {
                k: v for k, v in params.items() if k != "monotonic_cst"
            }
            model = HistGradientBoostingRegressor(
                **params_without_monotonic_cst,
                monotonic_cst=monotonic_cst[: data.shape[1]],
            )
        else:
            model = GradientBoostingRegressor(**params)
        model.fit(data, target)
        return model

    def _evaluate_model(
        self, models: List[Any], X_test: pd.DataFrame, y_test: pd.Series
    ) -> Dict[str, float]:
        inference_processor = InferenceProcessor(model_choice=self.model_choice)

        predictions = inference_processor.run(models, X_test)

        self._plot_predictions(y_test, predictions)

        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        mae = np.mean(np.abs(predictions - y_test))
        r2 = np.corrcoef(predictions, y_test)[0, 1] ** 2

        metrics = {"rmse": rmse, "mae": mae, "r2": r2}
        return metrics

    def _plot_predictions(self, y_test: pd.Series, predictions: np.ndarray) -> None:
        if self.path:
            import matplotlib.pyplot as plt

            plt.scatter(predictions, y_test, marker=".", color="black", s=10)
            plt.plot(
                [y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--", lw=2
            )
            plt.ylabel("True Values")
            plt.xlabel("Predictions")
            plt.title("True vs Predicted Values")
            plt.savefig(self.path)
            plt.close()
            logger.info(f"Predictions plot saved to {self.path}")

    def _generate_feature_importance_dataframe(
        self, model: Any, X_train: pd.DataFrame, y_train: pd.Series
    ) -> pd.DataFrame:
        if self.model_choice == "linear_regression":
            feature_importance_scores = model.coef_
        elif self.model_choice == "gam":
            feature_importance_scores = model.statistics_["p_values"]
            if np.all(feature_importance_scores == feature_importance_scores[0]):
                result = permutation_importance(
                    model, X_train, y_train, n_repeats=10, random_state=42
                )
                feature_importance_scores = result.importances_mean
        elif self.model_choice == "gbm":
            if isinstance(model, GradientBoostingRegressor):
                feature_importance_scores = model.feature_importances_
            elif isinstance(model, HistGradientBoostingRegressor):
                result = permutation_importance(
                    model, X_train, y_train, n_repeats=10, random_state=42
                )
                feature_importance_scores = result.importances_mean
            else:
                raise ValueError(
                    f"Model type {type(model)} not supported for feature importance extraction"
                )
        else:
            raise ValueError(f"Model choice {self.model_choice} not supported")

        feature_importance = self._formatting_feature_importance(
            feature_importance_scores
        )
        self._write_feature_importance_to_file(feature_importance)

        return feature_importance

    def _formatting_feature_importance(
        self, feature_importance_scores: np.ndarray
    ) -> pd.DataFrame:
        feature_importance = dict(
            zip(self.training_features, feature_importance_scores)
        )
        feature_importance = pd.DataFrame(
            list(feature_importance.items()), columns=["FEATURES", "IMPORTANCE"]
        )
        feature_importance = feature_importance.sort_values(
            by="IMPORTANCE", ascending=False
        )
        max_importance = feature_importance["IMPORTANCE"].max()
        feature_importance["IMPORTANCE"] = (
            feature_importance["IMPORTANCE"] / max_importance
        ) * 100

        return feature_importance

    def _write_feature_importance_to_file(
        self, feature_importance: pd.DataFrame
    ) -> None:
        if self.feature_importance_path:
            feature_importance.to_csv(self.feature_importance_path, index=False)
            logger.info(f"Feature importance written to {self.feature_importance_path}")
