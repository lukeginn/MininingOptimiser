import numpy as np
import logging as logger
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import ParameterGrid
from sklearn.linear_model import LinearRegression
from pygam import LinearGAM, s
from pygam.terms import TermList, SplineTerm
import joblib
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingRegressor
#import shap


def generate_model(
    data,
    target_feature,
    model_choice,
    param_grid,
    metric="rmse",
    training_features=None,
    generate_feature_importance=True,
    feature_importance_path=None,
    evaluation_results_path=None,
    test_size=0.2,
    random_state=42,
    n_models=5,
    path=None
):
    if training_features is None:
        logger.info("No training features provided. Using all numeric features")
        training_features = numeric_only_training_features(
            data=data, target_feature=target_feature
        )

    logger.info("Splitting data into target and features")
    data, target = split_data_into_target_and_training_features(
        data=data, target_feature=target_feature, training_features=training_features
    )

    logger.info("Splitting data into training and test sets")
    X_train, X_test, y_train, y_test = split_data(
        X=data, y=target, test_size=test_size, random_state=random_state
    )

    logger.info("Generating models")
    np.random.seed(random_state)
    models, best_params, best_rmse = find_best_hyperparameters(
        model_choice=model_choice,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        param_grid=param_grid,
        metric=metric,
        evaluation_results_path=evaluation_results_path,
        path=path,
        n_models=n_models
    )
    logger.info("Models generated successfully")

    if generate_feature_importance:
        logger.info("Generating feature importance")
        feature_importance = generate_feature_importance_dataframe(
            model_choice=model_choice,
            model=models[0],  # Use the first model for feature importance
            features=training_features,
            X_train=X_train,
            y_train=y_train,
            feature_importance_path=feature_importance_path,
        )
    else:
        feature_importance = None

    return models, best_params, best_rmse, feature_importance


def numeric_only_training_features(data, target_feature):
    return data.drop(columns=[target_feature]).select_dtypes(include="number").columns


def split_data_into_target_and_training_features(
    data, target_feature, training_features
):
    return data[training_features], data[target_feature]


def split_data(X, y, test_size, random_state):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    logger.info("Data split successfully")
    return X_train, X_test, y_train, y_test


def find_best_hyperparameters(
    model_choice, X_train, X_test, y_train, y_test, param_grid, metric, evaluation_results_path, path, n_models
):

    best_metric_score = float("inf")
    best_params = None
    best_models = []
    evaluation_results = []

    for params in ParameterGrid(param_grid):
        logger.info(f"Training {model_choice} model with parameters: {params}")
        training_function = get_model_training_function(model_choice=model_choice)
        models = [training_function(data=X_train, target=y_train, params=params) for _ in range(n_models)]

        logger.info(f"Evaluating {model_choice} models")
        metrics = evaluate_model(
            model_choice=model_choice, models=models, X_test=X_test, y_test=y_test, path=path
        )
        metric_score = metrics[metric]

        evaluation_results.append({**params, **metrics})

        if metric_score < best_metric_score:
            best_metric_score = metric_score
            best_params = params
            best_models = models

    export_evaluation_results(
        evaluation_results=pd.DataFrame(evaluation_results),
        path=evaluation_results_path
    )

    logger.info("Best hyperparameters found")
    logger.info(f"Best Parameters: {best_params}")
    logger.info(f"Best {metric}: {best_metric_score}")

    return best_models, best_params, best_metric_score

def export_evaluation_results(evaluation_results, path):
    evaluation_results.to_csv(path, index=False)
    logger.info(f"Evaluation results exported to {path}")

def get_model_training_function(model_choice):

    if model_choice == "linear_regression":

        def training_function(data, target, params):
            model = LinearRegression(**params)
            model.fit(data, target)
            return model

    if model_choice == "gam":

        def training_function(data, target, params):
            term_list = TermList()
            for i in range(data.shape[1]):
                constraint = params["constraints"][i]
                degree = int(params["degrees"][i])
                term = SplineTerm(i, constraints=constraint, n_splines=degree)
                term_list += term
            model = LinearGAM(term_list)
            model.fit(data, target)
            return model
        
    if model_choice == "gbm":

        def training_function(data, target, params):
            if "monotonic_cst" in params:
                monotonic_cst = params["monotonic_cst"]
                params_without_monotonic_cst = {k: v for k, v in params.items() if k != "monotonic_cst"}
                model = HistGradientBoostingRegressor(**params_without_monotonic_cst, monotonic_cst=monotonic_cst[:data.shape[1]])
            else:
                model = GradientBoostingRegressor(**params)
            model.fit(data, target)
            return model
        

    return training_function

def get_model_prediction_function(model_choice, return_confidence_interval=False, return_shapley_values=False):

    if model_choice == "linear_regression":

        def prediction_function(model, data):
            return model.predict(data)

    if model_choice == "gam":

        def prediction_function(model, data):
            if return_confidence_interval:
                prediction_interval = model.prediction_intervals(data, width=0.95)
                lower = prediction_interval[:, 0]
                upper = prediction_interval[:, 1]
                mean_prediction = model.predict(data)
                return lower, mean_prediction, upper
            if return_shapley_values:
                shapley_values = np.zeros(data.shape)
                for i in range(data.shape[1]):
                    shapley_values[:, i] = model.partial_dependence(term=i, X=data)
                shapley_values_df = pd.DataFrame(shapley_values, columns=data.columns)
                shapley_values_df['intercept'] =  model.predict(data) - shapley_values_df.sum(axis=1)
                shapley_values_df['final_prediction'] = model.predict(data)
                return shapley_values_df
            else:
                return model.predict(data)
            
    if model_choice == "gbm":
            
        def prediction_function(model, data):
            if return_confidence_interval:
            # GradientBoostingRegressor does not support prediction intervals natively
                raise NotImplementedError("Confidence intervals are not supported for GradientBoostingRegressor")
            if return_shapley_values:
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(data)
                shap_values_df = pd.DataFrame(shap_values, columns=data.columns)
                shap_values_df['final_prediction'] = model.predict(data)
                return shap_values_df
            else:
                return model.predict(data)

    return prediction_function


def evaluate_model(model_choice, models, X_test, y_test, path):
    prediction_function = get_model_prediction_function(model_choice=model_choice)
    predictions = np.mean([prediction_function(model=model, data=X_test) for model in models], axis=0)

    plot_predictions(y_test, predictions, path)

    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    mae = np.mean(np.abs(predictions - y_test))
    r2 = np.corrcoef(predictions, y_test)[0, 1] ** 2

    metrics = {
        "rmse": rmse,
        "mae": mae,
        "r2": r2
    }
    return metrics

def plot_predictions(y_test, predictions, path):
    import matplotlib.pyplot as plt
    plt.scatter(predictions, y_test, marker='.', color='black', s=10)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.ylabel("True Values")
    plt.xlabel("Predictions")
    plt.title("True vs Predicted Values")
    plt.savefig(path)
    plt.close()
    logger.info(f"Predictions plot saved to {path}")


def generate_predictions(model_choice, models, data, return_confidence_interval=False, return_shapley_values=False):
    prediction_function = get_model_prediction_function(
        model_choice, return_confidence_interval, return_shapley_values
    )
    predictions = np.mean([prediction_function(model, data) for model in models], axis=0)

    return predictions


def generate_feature_importance_dataframe(
    model_choice, model, features, X_train, y_train, feature_importance_path
):
    if model_choice == "linear_regression":
        feature_importance_scores = model.coef_
    elif model_choice == "gam":
        feature_importance_scores = model.statistics_["p_values"]
        if np.all(feature_importance_scores == feature_importance_scores[0]):
            # Use permutation importance as a fallback
            result = permutation_importance(model, X_train, y_train, n_repeats=10, random_state=42)
            feature_importance_scores = result.importances_mean
    elif model_choice == "gbm":
        if isinstance(model, GradientBoostingRegressor):
            feature_importance_scores = model.feature_importances_
        elif isinstance(model, HistGradientBoostingRegressor):
            result = permutation_importance(model, X_train, y_train, n_repeats=10, random_state=42)
            feature_importance_scores = result.importances_mean
        else:
            raise ValueError(f"Model type {type(model)} not supported for feature importance extraction")
        
    else:
        raise ValueError(f"Model choice {model_choice} not supported")

    feature_importance = formatting_feature_importance(
        features=features, feature_importance_scores=feature_importance_scores
    )

    write_feature_importance_to_file(
        feature_importance=feature_importance, path=feature_importance_path
    )

    return feature_importance


def formatting_feature_importance(features, feature_importance_scores):
    feature_importance = dict(zip(features, feature_importance_scores))
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


def write_feature_importance_to_file(feature_importance, path):
    feature_importance.to_csv(path, index=False)
    logger.info(f"Feature importance written to {path}")


def save_models(models, path, path_suffix):
    path = str(path)
    for i, model in enumerate(models):
        full_file_path = f"{path}\\model_{path_suffix}_{i}.pkl"
        joblib.dump(model, full_file_path)
        logger.info(f"Model saved to {full_file_path}")


def load_models(file_paths):
    models = [joblib.load(file_path) for file_path in file_paths]
    logger.info(f"Models loaded from {file_paths}")
    return models
