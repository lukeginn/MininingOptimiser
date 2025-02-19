import config.paths as paths
from src.data.reading.functions.read_csv import read_csv
from src.data.preprocessing.functions.initial_preprocessing import initial_preprocessing
from src.utils.generate_artifacts import generate_artifacts

class DataReader:
    def __init__(self, config):
        self.config = config

    def read_file(self):
        data = read_csv(file_path=paths.Paths.DATA_FILE_1.value)
        data = initial_preprocessing(data)
        self.generate_artifacts_for_read_file(data)
        return data

    def generate_artifacts_for_read_file(self, data):
        paths_dict = {
            'time_series_plots': paths.Paths.TIME_SERIES_PLOTS_FOR_RAW_DATA_PATH.value,
            'histogram_plots': paths.Paths.HISTOGRAM_PLOTS_FOR_RAW_DATA_PATH.value,
            'custom_plots': paths.Paths.CUSTOM_PLOTS_FOR_RAW_DATA_PATH.value
        }
        generate_artifacts(self.config, data, "stage_1_data_reading", paths_dict)