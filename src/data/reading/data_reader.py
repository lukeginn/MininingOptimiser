import config.paths as paths
from shared.data.export_data import export_data
from shared.data.generate_meta_data import generate_meta_data
from shared.data.generate_time_series_plots import generate_time_series_plots
from shared.data.generate_histogram_plots import generate_histogram_plots
from src.visualisation.custom_plots import custom_plots
from src.data.reading.read_csv import read_csv
from src.data.preprocessing.initial_preprocessing import initial_preprocessing

class DataReader:
    def __init__(self, config):
        self.config = config

    def read_file(self):
        data = read_csv(file_path=paths.Paths.DATA_FILE_1.value)
        data = initial_preprocessing(data)
        self.generate_artifacts(data)
        return data

    def generate_artifacts(self, data):
        if self.config.export_data:
            export_data(
                data=data,
                path=paths.Paths.EXPORTED_DATA_FILE.value,
                path_suffix="stage_1_data_reading",
            )
        if self.config.generate_meta_data:
            generate_meta_data(
                data=data,
                path=paths.Paths.META_DATA_FILE.value,
                path_suffix="stage_1_data_reading",
            )
        if self.config.generate_time_series_plots:
            generate_time_series_plots(
                data=data,
                timestamp=self.config.data.timestamp,
                path=paths.Paths.TIME_SERIES_PLOTS_FOR_RAW_DATA_PATH.value,
            )
        if self.config.generate_histogram_plots.run:
            generate_histogram_plots(
                data=data,
                path=paths.Paths.HISTOGRAM_PLOTS_FOR_RAW_DATA_PATH.value,
                number_of_bins=self.config.generate_histogram_plots.number_of_bins,
            )
        if self.config.generate_custom_plots:
            custom_plots(
                data=data,
                path=paths.Paths.CUSTOM_PLOTS_FOR_RAW_DATA_PATH.value
            )