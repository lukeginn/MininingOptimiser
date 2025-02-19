import config.paths as paths
from src.utils.generate_artifacts import generate_artifacts

class ShutdownFilter:
    def __init__(self, config):
        self.config = config

    def run(self, data):
        if self.config.data.filter_shutdowns.run:   
            data = self.filter_shutdowns(data)
            self.generate_artifacts_for_run(data)
        return data
    
    def generate_artifacts_for_run(self, data):
        paths_dict = {
            'time_series_plots': paths.Paths.TIME_SERIES_PLOTS_FOR_FILTERING_SHUTDOWN_PATH.value,
            'histogram_plots': paths.Paths.HISTOGRAM_PLOTS_FOR_FILTERING_SHUTDOWN_PATH.value,
            'custom_plots': paths.Paths.CUSTOM_PLOTS_FOR_FILTERING_SHUTDOWN_PATH.value
        }
        generate_artifacts(self.config, data, "stage_10_filter_shutdowns", paths_dict)

    def filter_shutdowns(self, data):
        shutdown_dates = self.identify_shutdown_times(data)
        if shutdown_dates is not None:
            data = data[~data['DATE'].isin(shutdown_dates)]
        return data
    
    def identify_shutdown_times(self, data):
        # Currently no shutdown periods have been identified
        shutdown_dates = None
        return shutdown_dates

