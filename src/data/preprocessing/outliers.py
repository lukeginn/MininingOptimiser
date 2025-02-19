import config.paths as paths
from shared.data.export_data import export_data
from shared.data.generate_meta_data import generate_meta_data
from shared.data.generate_time_series_plots import generate_time_series_plots
from shared.data.generate_histogram_plots import generate_histogram_plots
from src.visualisation.custom_plots import custom_plots
from shared.data.outlier_identifier import identify_outliers

def identifying_outliers(data, config):
    if config.data.identify_outliers.run:
        data = identify_outliers(
            data=data,
            method=config.data.identify_outliers.method,
            iqr_threshold=config.data.identify_outliers.iqr_threshold,
            z_score_threshold=config.data.identify_outliers.z_score_threshold,
            mad_threshold=config.data.identify_outliers.mad_threshold,
            dbscan_eps=config.data.identify_outliers.dbscan_eps,
            dbscan_min_samples=config.data.identify_outliers.dbscan_min_samples,
            isolation_forest_threshold=config.data.identify_outliers.isolation_forest_threshold,
            lof_threshold=config.data.identify_outliers.lof_threshold,
        )
        if config.export_data:
            export_data(
                data=data,
                path=paths.Paths.EXPORTED_DATA_FILE.value,
                path_suffix="stage_3_outliers_identified",
            )
        if config.generate_meta_data:
            generate_meta_data(
                data=data,
                path=paths.Paths.META_DATA_FILE.value,
                path_suffix="stage_3_outliers_identified",
            )
        if config.generate_time_series_plots:
            generate_time_series_plots(
                data=data,
                timestamp=config.data.timestamp,
                path=paths.Paths.TIME_SERIES_PLOTS_FOR_OUTLIERS_IDENTIFIED_PATH.value,
            )
        if config.generate_histogram_plots.run:
            generate_histogram_plots(
                data=data,
                path=paths.Paths.HISTOGRAM_PLOTS_FOR_OUTLIERS_IDENTIFIED_PATH.value,
                number_of_bins=config.generate_histogram_plots.number_of_bins,
            )
        if config.generate_custom_plots:
            custom_plots(
                data=data,
                path=paths.Paths.CUSTOM_PLOTS_FOR_OUTLIERS_IDENTIFIED_PATH.value,
            )
    return data