import numpy as np
import logging as logger
import config.paths as paths
from shared.utils.config import read_config

def setup():
    logger.info("Setting up paths and configurations")
    paths.create_directories()
    config = read_config(config_file_path=paths.Paths.CONFIG_FILE_PATH.value)
    np.random.seed(config.random_state)
    return config