import numpy as np
import pandas as pd
import logging as logger


def generate_data(num_rows, num_features):

    np.random.seed(42)
    data = pd.DataFrame(
        np.random.rand(num_rows, num_features),
        columns=[f"feature{i+1}" for i in range(num_features)],
    )
    data = data.div(data.sum(axis=1), axis=0)
    correlated_features = data.columns[: num_features // 2]
    data["recovery"] = data[correlated_features].mean(axis=1) + np.random.normal(
        0, 0.5, num_rows
    )
    data["recovery"] = (data["recovery"] - data["recovery"].min()) / (
        data["recovery"].max() - data["recovery"].min()
    )
    logger.info("Data generated successfully")

    return data
