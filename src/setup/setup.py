import numpy as np
import logging as logger
import config.paths as paths
from shared.utils.config import read_config


def setup():
    logger.info("Setting up paths and configurations")
    paths.create_directories()
    general_config = read_config(
        config_file_path=paths.Paths.GENERAL_CONFIG_FILE_PATH.value
    )
    data_config = read_config(config_file_path=paths.Paths.DATA_CONFIG_FILE_PATH.value)
    model_config = read_config(
        config_file_path=paths.Paths.MODEL_CONFIG_FILE_PATH.value
    )
    clustering_config = read_config(
        config_file_path=paths.Paths.CLUSTERING_CONFIG_FILE_PATH.value
    )
    simulation_config = read_config(
        config_file_path=paths.Paths.SIMULATION_CONFIG_FILE_PATH.value
    )
    optimisation_config = read_config(
        config_file_path=paths.Paths.OPTIMISATION_CONFIG_FILE_PATH.value
    )
    np.random.seed(general_config.random_state)
    return (
        general_config,
        data_config,
        model_config,
        clustering_config,
        simulation_config,
        optimisation_config,
    )
