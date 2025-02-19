import config.paths as paths
from shared.data.export_data import export_data
from shared.data.generate_meta_data import generate_meta_data
from shared.data.generate_time_series_plots import generate_time_series_plots
from shared.data.generate_histogram_plots import generate_histogram_plots
from src.data.custom_plots import custom_plots
from shared.data.missing_data_identifier import identify_missing_data
from shared.data.missing_data_correction import missing_data_correction

def identifying_missing_data(data, config):
    if config.data.identify_missing_data.run:
        data = identify_missing_data(
            data=data,
            timestamp=config.data.timestamp,
            unique_values_identification=config.data.identify_missing_data.unique_values.run,
            unique_values_threshold=config.data.identify_missing_data.unique_values.threshold,
            explicit_missing_values=config.data.identify_missing_data.explicit_missing_values.run,
            explicit_missing_indicators=config.data.identify_missing_data.explicit_missing_values.indicators,
            repeating_values=config.data.identify_missing_data.repeating_values.run,
            repeating_values_threshold=config.data.identify_missing_data.repeating_values.threshold,
            repeating_values_proportion_threshold=config.data.identify_missing_data.repeating_values.proportion_threshold,
        )
        if config.export_data:
            export_data(
                data=data,
                path=paths.Paths.EXPORTED_DATA_FILE.value,
                path_suffix="stage_2_missing_data_identified",
            )
        if config.generate_meta_data:
            generate_meta_data(
                data=data,
                path=paths.Paths.META_DATA_FILE.value,
                path_suffix="stage_2_missing_data_identified",
            )
        if config.generate_time_series_plots:
            generate_time_series_plots(
                data=data,
                timestamp=config.data.timestamp,
                path=paths.Paths.TIME_SERIES_PLOTS_FOR_MISSING_DATA_IDENTIFIED_PATH.value,
            )
        if config.generate_histogram_plots.run:
            generate_histogram_plots(
                data=data,
                path=paths.Paths.HISTOGRAM_PLOTS_FOR_MISSING_DATA_IDENTIFIED_PATH.value,
                number_of_bins=config.generate_histogram_plots.number_of_bins,
            )
        if config.generate_custom_plots:
            custom_plots(
                data=data,
                path=paths.Paths.CUSTOM_PLOTS_FOR_MISSING_DATA_IDENTIFIED_PATH.value,
            )
    return data

def correcting_missing_data(data, config):
    if config.data.correct_missing_data.run:
        data = missing_data_correction(
            data=data,
            delete_all_rows_with_missing_values=config.data.correct_missing_data.delete_all_rows_with_missing_values.run,
            interpolate_time_series=config.data.correct_missing_data.interpolate_time_series.run,
            interpolate_highly_regular_time_series=config.data.correct_missing_data.interpolate_highly_regular_time_series.run,
            replace_missing_values_with_x=config.data.correct_missing_data.replace_missing_values_with_x.run,
            replace_missing_values_with_last_known_value=config.data.correct_missing_data.replace_missing_values_with_last_known_value.run,
            interpolate_method=config.data.correct_missing_data.interpolate_time_series.method,
            interpolate_limit_direction=config.data.correct_missing_data.interpolate_time_series.limit_direction,
            interpolate_max_gap=config.data.correct_missing_data.interpolate_time_series.max_gap,
            interpolate_highly_regular_method=config.data.correct_missing_data.interpolate_highly_regular_time_series.method,
            interpolate_highly_regular_limit_direction=config.data.correct_missing_data.interpolate_highly_regular_time_series.limit_direction,
            interpolate_highly_regular_interval_min=config.data.correct_missing_data.interpolate_highly_regular_time_series.regular_interval_min,
            replace_missing_values_with_last_known_value_backfill=config.data.correct_missing_data.replace_missing_values_with_last_known_value.backfill,
            timestamp=config.data.timestamp,
            x=config.data.correct_missing_data.replace_missing_values_with_x.x,
        )
        if config.export_data:
            export_data(
                data=data,
                path=paths.Paths.EXPORTED_DATA_FILE.value,
                path_suffix="stage_5_missing_data_corrected",
            )
        if config.generate_meta_data:
            generate_meta_data(
                data=data,
                path=paths.Paths.META_DATA_FILE.value,
                path_suffix="stage_5_missing_data_corrected",
            )
        if config.generate_time_series_plots:
            generate_time_series_plots(
                data=data,
                timestamp=config.data.timestamp,
                path=paths.Paths.TIME_SERIES_PLOTS_FOR_MISSING_DATA_CORRECTED_PATH.value,
            )
        if config.generate_histogram_plots.run:
            generate_histogram_plots(
                data=data,
                path=paths.Paths.HISTOGRAM_PLOTS_FOR_MISSING_DATA_CORRECTED_PATH.value,
                number_of_bins=config.generate_histogram_plots.number_of_bins,
            )
        if config.generate_custom_plots:
            custom_plots(
                data=data,
                path=paths.Paths.CUSTOM_PLOTS_FOR_MISSING_DATA_CORRECTED_PATH.value
            )
    return data

def correcting_missing_data_post_aggregation(data, config):
    if config.data.correct_missing_data_after_aggregation.run:
        data = missing_data_correction(
            data=data,
            delete_all_rows_with_missing_values=config.data.correct_missing_data_after_aggregation.delete_all_rows_with_missing_values.run,
            interpolate_time_series=config.data.correct_missing_data_after_aggregation.interpolate_time_series.run,
            interpolate_highly_regular_time_series=config.data.correct_missing_data_after_aggregation.interpolate_highly_regular_time_series.run,
            replace_missing_values_with_x=config.data.correct_missing_data_after_aggregation.replace_missing_values_with_x.run,
            replace_missing_values_with_last_known_value=config.data.correct_missing_data_after_aggregation.replace_missing_values_with_last_known_value.run,
            interpolate_method=config.data.correct_missing_data_after_aggregation.interpolate_time_series.method,
            interpolate_limit_direction=config.data.correct_missing_data_after_aggregation.interpolate_time_series.limit_direction,
            interpolate_max_gap=config.data.correct_missing_data_after_aggregation.interpolate_time_series.max_gap,
            interpolate_highly_regular_method=config.data.correct_missing_data_after_aggregation.interpolate_highly_regular_time_series.method,
            interpolate_highly_regular_limit_direction=config.data.correct_missing_data_after_aggregation.interpolate_highly_regular_time_series.limit_direction,
            interpolate_highly_regular_interval_min=config.data.correct_missing_data_after_aggregation.interpolate_highly_regular_time_series.regular_interval_min,
            replace_missing_values_with_last_known_value_backfill=config.data.correct_missing_data_after_aggregation.replace_missing_values_with_last_known_value.backfill,
            timestamp=config.data.timestamp,
            x=config.data.correct_missing_data_after_aggregation.replace_missing_values_with_x.x,
        )
        if config.export_data:
            export_data(
                data=data,
                path=paths.Paths.EXPORTED_DATA_FILE.value,
                path_suffix="stage_9_missing_data_corrected_post_aggregation",
            )
        if config.generate_meta_data:
            generate_meta_data(
                data=data,
                path=paths.Paths.META_DATA_FILE.value,
                path_suffix="stage_9_missing_data_corrected_post_aggregation",
            )
        if config.generate_time_series_plots:
            generate_time_series_plots(
                data=data,
                timestamp=config.data.timestamp,
                path=paths.Paths.TIME_SERIES_PLOTS_FOR_MISSING_DATA_CORRECTED_POST_AGGREGATION_PATH.value,
            )
        if config.generate_histogram_plots.run:
            generate_histogram_plots(
                data=data,
                path=paths.Paths.HISTOGRAM_PLOTS_FOR_MISSING_DATA_CORRECTED_POST_AGGREGATION_PATH.value,
                number_of_bins=config.generate_histogram_plots.number_of_bins,
            )
        if config.generate_custom_plots:
            custom_plots(
                data=data,
                path=paths.Paths.CUSTOM_PLOTS_FOR_MISSING_DATA_CORRECTED_POST_AGGREGATION_PATH.value
            )
    return data