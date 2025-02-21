import pandas as pd
import logging as logger


def generate_meta_data(data, path, features=None, path_suffix=None):
    if features is None:
        features = data.columns
    meta_data = _extract_meta_data(data, features)
    _write_to_file(meta_data, path, path_suffix)

    return meta_data


def _extract_meta_data(data, features):
    numeric_features = data[features].select_dtypes(include=["number"]).columns
    timestamp_features = data[features].select_dtypes(include=["datetime"]).columns
    timestamp_and_numeric_features = list(timestamp_features) + list(numeric_features)

    meta_data = {}
    meta_data["Data: Column Count"] = data.shape[1]
    meta_data["Data: Row Count"] = data.shape[0]
    meta_data["Feature: Column Type"] = (
        data[features]
        .dtypes.apply(lambda x: "str" if x == "object" else x.name)
        .to_dict()
    )
    meta_data["Feature: Non-Missing Values Count"] = data[features].count().to_dict()
    meta_data["Feature: Missing Values Count"] = data[features].isnull().sum().to_dict()
    meta_data["Feature: Unique Values Count"] = data[features].nunique().to_dict()
    meta_data["Feature: Mean"] = data[numeric_features].mean().to_dict()
    meta_data["Feature: Minimum"] = data[timestamp_and_numeric_features].min().to_dict()
    quantiles = data[timestamp_and_numeric_features].quantile([0.25])
    for quantile in quantiles.index:
        meta_data[f"Feature: Quantile {int(quantile*100)}"] = quantiles.loc[
            quantile
        ].to_dict()
    meta_data["Feature: Median"] = (
        data[timestamp_and_numeric_features].median().to_dict()
    )
    quantiles = data[timestamp_and_numeric_features].quantile([0.75])
    for quantile in quantiles.index:
        meta_data[f"Feature: Quantile {int(quantile*100)}"] = quantiles.loc[
            quantile
        ].to_dict()
    meta_data["Feature: Maximum"] = data[timestamp_and_numeric_features].max().to_dict()
    meta_data["Feature: Standard Deviation"] = data[numeric_features].std().to_dict()
    meta_data["Feature: Variance"] = data[numeric_features].var().to_dict()
    meta_data["Feature: Skewness"] = data[numeric_features].skew().to_dict()
    meta_data["Feature: Kurtosis"] = data[numeric_features].kurtosis().to_dict()
    meta_data["Feature: Mode"] = data[numeric_features].mode().iloc[0].to_dict()
    logger.info("Meta data extracted successfully")

    return meta_data


def _write_to_file(meta_data, path, path_suffix):
    path = str(path)
    path = path.replace(".csv", "")
    output_df = pd.DataFrame(meta_data).T
    output_df.index.name = "Meta Data"
    full_file_path = f"{path}_{path_suffix}.csv"
    output_df.to_csv(full_file_path, index=True)
    logger.info(f"Meta data written to {full_file_path}")

    return None
