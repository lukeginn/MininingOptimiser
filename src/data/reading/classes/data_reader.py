import config.paths as paths
from src.data.reading.functions.read_csv import read_csv
from src.data.preprocessing.functions.initial_preprocessing import initial_preprocessing
from src.utils.generate_artifacts import generate_artifacts
from dataclasses import dataclass
from typing import Dict, Any
import pandas as pd


@dataclass
class DataReader:
    general_config: Dict[str, Any]
    data_config: Dict[str, Any]

    def run(self) -> pd.DataFrame:
        data = read_csv(file_path=paths.Paths.DATA_FILE_1.value)
        data = initial_preprocessing(data)
        self.generate_artifacts_for_read_file(data)
        return data

    def generate_artifacts_for_read_file(self, data: pd.DataFrame) -> None:
        paths_dict = {
            "time_series_plots": paths.Paths.TIME_SERIES_PLOTS_FOR_RAW_DATA_PATH.value,
            "histogram_plots": paths.Paths.HISTOGRAM_PLOTS_FOR_RAW_DATA_PATH.value,
            "custom_plots": paths.Paths.CUSTOM_PLOTS_FOR_RAW_DATA_PATH.value,
        }
        generate_artifacts(
            self.general_config,
            self.data_config,
            data,
            "stage_1_data_reading",
            paths_dict,
        )
