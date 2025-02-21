import config.paths as paths
from shared.data.functions.export_data import export_data
from shared.data.functions.generate_meta_data import generate_meta_data
from shared.data.functions.generate_time_series_plots import generate_time_series_plots
from shared.data.functions.generate_histogram_plots import generate_histogram_plots
from src.visualisation.custom_plots import custom_plots


def generate_artifacts(general_config, data_config, data, stage_suffix, paths_dict):
    if general_config.export_data:
        export_data(
            data=data,
            path=paths.Paths.EXPORTED_DATA_FILE.value,
            path_suffix=stage_suffix,
        )
    if general_config.generate_meta_data:
        generate_meta_data(
            data=data,
            path=paths.Paths.META_DATA_FILE.value,
            path_suffix=stage_suffix,
        )
    if general_config.generate_time_series_plots:
        generate_time_series_plots(
            data=data,
            timestamp=data_config.timestamp,
            path=paths_dict["time_series_plots"],
        )
    if general_config.generate_histogram_plots.run:
        generate_histogram_plots(
            data=data,
            path=paths_dict["histogram_plots"],
            number_of_bins=general_config.generate_histogram_plots.number_of_bins,
        )
    if general_config.generate_custom_plots:
        custom_plots(data=data, path=paths_dict["custom_plots"])