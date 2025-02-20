import numpy as np
import logging as logger
import config.paths as paths
from shared.utils.config import read_config
from dataclasses import dataclass, field
from typing import Dict


@dataclass
class Setup:
    general_config: Dict[str, any] = field(init=False)
    data_config: Dict[str, any] = field(init=False)
    model_config: Dict[str, any] = field(init=False)
    clustering_config: Dict[str, any] = field(init=False)
    simulation_config: Dict[str, any] = field(init=False)
    optimisation_config: Dict[str, any] = field(init=False)

    def __post_init__(self) -> None:
        logger.info("Setting up paths and configurations")
        paths.create_directories()
        self.general_config = read_config(
            config_file_path=paths.Paths.GENERAL_CONFIG_FILE_PATH.value
        )
        self.data_config = read_config(
            config_file_path=paths.Paths.DATA_CONFIG_FILE_PATH.value
        )
        self.model_config = read_config(
            config_file_path=paths.Paths.MODEL_CONFIG_FILE_PATH.value
        )
        self.clustering_config = read_config(
            config_file_path=paths.Paths.CLUSTERING_CONFIG_FILE_PATH.value
        )
        self.simulation_config = read_config(
            config_file_path=paths.Paths.SIMULATION_CONFIG_FILE_PATH.value
        )
        self.optimisation_config = read_config(
            config_file_path=paths.Paths.OPTIMISATION_CONFIG_FILE_PATH.value
        )
        np.random.seed(self.general_config.random_state)
