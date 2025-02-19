import config.paths as paths
from shared.data.outlier_identifier import identify_outliers
from src.utils.generate_artifacts import generate_artifacts

class OutlierProcessor:
    def __init__(self, config):
        self.config = config

    def run(self, data):
        if self.config.data.identify_outliers.run:
            data = identify_outliers(
                data=data,
                method=self.config.data.identify_outliers.method,
                iqr_threshold=self.config.data.identify_outliers.iqr_threshold,
                z_score_threshold=self.config.data.identify_outliers.z_score_threshold,
                mad_threshold=self.config.data.identify_outliers.mad_threshold,
                dbscan_eps=self.config.data.identify_outliers.dbscan_eps,
                dbscan_min_samples=self.config.data.identify_outliers.dbscan_min_samples,
                isolation_forest_threshold=self.config.data.identify_outliers.isolation_forest_threshold,
                lof_threshold=self.config.data.identify_outliers.lof_threshold,
            )
            self.generate_artifacts_for_identifying_outliers(data)
        return data
    
    def generate_artifacts_for_identifying_outliers(self, data):
        paths_dict = {
            'time_series_plots': paths.Paths.TIME_SERIES_PLOTS_FOR_OUTLIERS_IDENTIFIED_PATH.value,
            'histogram_plots': paths.Paths.HISTOGRAM_PLOTS_FOR_OUTLIERS_IDENTIFIED_PATH.value,
            'custom_plots': paths.Paths.CUSTOM_PLOTS_FOR_OUTLIERS_IDENTIFIED_PATH.value
        }
        generate_artifacts(self.config, data, "stage_3_outliers_identified", paths_dict)