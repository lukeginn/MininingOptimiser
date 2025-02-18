import logging as logger
from sklearn.cluster import DBSCAN
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor


def identify_outliers(
    data,
    features=None,
    method="mad",
    iqr_threshold=1.5,
    z_score_threshold=3,
    mad_threshold=3,
    dbscan_eps=0.5,
    dbscan_min_samples=5,
    isolation_forest_threshold=0.001,
    lof_threshold=0.001,
):
    logger.info("Starting outlier identification process.")
    logger.info("Method to be used: %s", method)

    if features is None:
        features = numerical_features(data=data)

    for feature in features:
        logger.info(f"Processing feature: {feature}")
        if method == "iqr":
            # Pros: Simple and effective for many datasets.
            # Cons: May not be effective for datasets with skewed distributions.
            data = identify_outliers_via_iqr(
                data=data, feature=feature, iqr_threshold=iqr_threshold
            )
        elif method == "z_score":
            # Pros: Effective for normally distributed data.
            # Cons: Not suitable for non-normal distributions.
            data = identify_outliers_via_z_score(
                data=data, feature=feature, z_score_threshold=z_score_threshold
            )
        elif method == "mad":
            # Pros: More robust to non-normal distributions and outliers.
            # Cons: More complex to compute.
            data = identify_outliers_via_mad(
                data=data, feature=feature, mad_threshold=mad_threshold
            )
        elif method == "dbscan":
            # Pros: Effective for datasets with clusters of varying shapes and sizes.
            # Cons: Requires tuning and more computationally intensive.
            data = identify_outliers_via_dbscan(
                data=data,
                feature=feature,
                dbscan_eps=dbscan_eps,
                dbscan_min_samples=dbscan_min_samples,
            )
        elif method == "isolation_forest":
            # Pros: Effective for high-dimensional datasets.
            # Cons: Requires tuning and more computationally intensive.
            data = identify_outliers_via_isolation_forest(
                data=data,
                feature=feature,
                isolation_forest_threshold=isolation_forest_threshold,
            )
        elif method == "lof":
            # Pros: Effective for datasets with varying density.
            # Cons: Requires tuning and more computationally intensive.
            data = identify_outliers_via_lof(
                data=data, feature=feature, lof_threshold=lof_threshold
            )

    return data


def numerical_features(data):
    return data.select_dtypes(include="number").columns


def identify_outliers_via_iqr(data, feature, iqr_threshold=1.5):
    Q1 = data[feature].quantile(0.25)
    Q3 = data[feature].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - iqr_threshold * IQR
    upper_bound = Q3 + iqr_threshold * IQR
    data[feature] = data[feature].mask(
        (data[feature] < lower_bound) | (data[feature] > upper_bound)
    )
    return data


def identify_outliers_via_z_score(data, feature, z_score_threshold=3):
    z_scores = (data[feature] - data[feature].mean()) / data[feature].std()
    data[feature] = data[feature].mask(abs(z_scores) > z_score_threshold)
    return data


def identify_outliers_via_mad(data, feature, mad_threshold=3):
    if data[feature].nunique() < 2:
        logger.warning(
            f"Feature {feature} has less than 2 unique values. Skipping MAD outlier detection."
        )
        return data
    median = data[feature].median()
    mad = (data[feature] - median).abs().median()
    if mad == 0:
        logger.warning(
            f"Feature {feature} has a MAD of 0. Skipping MAD outlier detection."
        )
        return data
    data[feature] = data[feature].mask(
        abs(data[feature] - median) / mad > mad_threshold
    )
    return data


def identify_outliers_via_dbscan(data, feature, dbscan_eps, dbscan_min_samples):
    dbscan = DBSCAN(eps=dbscan_eps, min_samples=dbscan_min_samples)
    data_copy = data.ffill().bfill()
    outliers = dbscan.fit_predict(data_copy[feature].values.reshape(-1, 1))
    data[feature] = data[feature].mask(outliers == -1)
    return data


def identify_outliers_via_isolation_forest(data, feature, isolation_forest_threshold):
    if not (0.0 < isolation_forest_threshold <= 0.5):
        logger.warning(
            f"Threshold for IsolationForest must be in the range (0.0, 0.5]. Got {isolation_forest_threshold}. Setting to default 0.05."
        )
        isolation_forest_threshold = 0.001
    isolation_forest = IsolationForest(contamination=isolation_forest_threshold)
    data_copy = data.ffill().bfill()
    outliers = isolation_forest.fit_predict(data_copy[feature].values.reshape(-1, 1))
    data[feature] = data[feature].mask(outliers == -1)
    return data


def identify_outliers_via_lof(data, feature, lof_threshold):

    if not (0.0 < lof_threshold <= 0.5):
        logger.warning(
            f"Threshold for LOF must be in the range (0.0, 0.5]. Got {lof_threshold}. Setting to default 0.05."
        )
        lof_threshold = 0.001
    lof = LocalOutlierFactor(n_neighbors=20, contamination=lof_threshold)
    data_copy = data.ffill().bfill()
    outliers = lof.fit_predict(data_copy[feature].values.reshape(-1, 1))
    data[feature] = data[feature].mask(outliers == -1)
    return data
