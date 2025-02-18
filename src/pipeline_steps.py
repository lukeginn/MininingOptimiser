import numpy as np
import logging as logger
import config.paths as paths
from shared.utils.config import read_config
from shared.data.generate_meta_data import generate_meta_data
from shared.model.generate_model import generate_model, save_models
from shared.model.generate_partial_plots import generate_partial_plots
from shared.data.aggregate_data import rolling_aggregate_data_via_timestamp
from shared.data.generate_time_series_plots import generate_time_series_plots
from shared.data.generate_histogram_plots import generate_histogram_plots
from shared.data.missing_data_correction import missing_data_correction
from shared.data.missing_data_identifier import identify_missing_data
from src.data.shutdown_identifier import identify_shutdowns
from shared.data.introduce_lags import introduce_lags
from shared.data.outlier_identifier import identify_outliers
from shared.data.export_data import export_data
from shared.model.feature_selection import feature_selection
from shared.model.univariable_feature_importance import (
    generate_univariable_feature_importance,
)
from shared.model.generate_correlation_matrix import generate_correlation_matrix
from shared.model.generate_clusters import run_clustering
from src.data.data_cache import cache_data
from src.data.generate_simulation import generate_simulations
from src.data.read_csv import read_csv
from src.data.preprocess_data import preprocess_dataset
from src.data.feature_engineering import feature_engineering
from src.data.custom_plots import custom_plots
from src.model.cluster_combinations import create_cluster_combinations
from src.model.optimal_clusters import create_optimised_clusters

def setup():
    logger.info("Setting up paths and configurations")
    paths.create_directories()
    config = read_config(config_file_path=paths.Paths.CONFIG_FILE_PATH.value)
    np.random.seed(config.random_state)
    return config

def read_data():
    data = read_csv(file_path=paths.Paths.DATA_FILE_1.value)
    return data

def preprocess_data(data):
    data = preprocess_dataset(data)
    return data

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

def identifying_shutdowns(data, config):
    if config.data.identify_shutdowns.run:
        data = identify_shutdowns(
            data=data,
            shutdown_features=config.data.identify_shutdowns.shutdown_features,
            cutoff_values=config.data.identify_shutdowns.cutoff_values,
        )
        if config.export_data:
            export_data(
                data=data,
                path=paths.Paths.EXPORTED_DATA_FILE.value,
                path_suffix="stage_4_shutdowns_identified",
            )
        if config.generate_meta_data:
            generate_meta_data(
                data=data,
                path=paths.Paths.META_DATA_FILE.value,
                path_suffix="stage_4_shutdowns_identified",
            )
        if config.generate_time_series_plots:
            generate_time_series_plots(
                data=data,
                timestamp=config.data.timestamp,
                path=paths.Paths.TIME_SERIES_PLOTS_FOR_SHUTDOWNS_IDENTIFIED_PATH.value
            )
        if config.generate_histogram_plots.run:
            generate_histogram_plots(
                data=data,
                path=paths.Paths.HISTOGRAM_PLOTS_FOR_SHUTDOWNS_IDENTIFIED_PATH.value,
                number_of_bins=config.generate_histogram_plots.number_of_bins,
            )
        if config.generate_custom_plots:
            custom_plots(
                data=data,
                path=paths.Paths.CUSTOM_PLOTS_FOR_SHUTDOWNS_IDENTIFIED_PATH.value,
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

def introducing_lags(data, config):
    if config.data.introduce_lags.run:
        data = introduce_lags(
            data=data,
            timestamp=config.data.timestamp,
            features=config.data.introduce_lags.features,
            lags=config.data.introduce_lags.lags,
            optimise_lags=config.data.introduce_lags.optimise_lags,
            target=config.data.introduce_lags.target,
            max_lag=config.data.introduce_lags.max_lag,
            overwrite_existing_features=config.data.introduce_lags.overwrite_existing_features,
        )
        if config.export_data:
            export_data(
                data=data,
                path=paths.Paths.EXPORTED_DATA_FILE.value,
                path_suffix="stage_6_lags_introduced",
            )
        if config.generate_meta_data:
            generate_meta_data(
                data=data,
                path=paths.Paths.META_DATA_FILE.value,
                path_suffix="stage_6_lags_introduced",
            )
        if config.generate_time_series_plots:
            generate_time_series_plots(
                data=data,
                timestamp=config.data.timestamp,
                path=paths.Paths.TIME_SERIES_PLOTS_FOR_LAGGED_FEATURES_PATH.value,
            )
        if config.generate_histogram_plots.run:
            generate_histogram_plots(
                data=data,
                path=paths.Paths.HISTOGRAM_PLOTS_FOR_LAGGED_FEATURES_PATH.value,
                number_of_bins=config.generate_histogram_plots.number_of_bins,
            )
        if config.generate_custom_plots:
            custom_plots(
                data=data,
                path=paths.Paths.CUSTOM_PLOTS_FOR_LAGGED_FEATURES_PATH.value
            )
    return data

def aggregating_data(data, config):
    data = rolling_aggregate_data_via_timestamp(
        data=data,
        timestamp=config.data.timestamp,
        aggregation_types=config.data.rolling_aggregation.aggregation_types,
        window=config.data.rolling_aggregation.window,
        min_periods=config.data.rolling_aggregation.min_periods,
        window_selection_frequency=config.data.rolling_aggregation.window_selection_frequency,
    )
    if config.export_data:
        export_data(
            data=data,
            path=paths.Paths.EXPORTED_DATA_FILE.value,
            path_suffix="stage_7_rolling_aggregate_data",
        )
    if config.generate_meta_data:
        generate_meta_data(
            data=data,
            path=paths.Paths.META_DATA_FILE.value,
            path_suffix="stage_7_rolling_aggregate_data",
        )
    if config.generate_time_series_plots:
        generate_time_series_plots(
            data=data,
            timestamp=config.data.timestamp,
            path=paths.Paths.TIME_SERIES_PLOTS_FOR_AGGREGATED_FEATURES_PATH.value,
        )
    if config.generate_histogram_plots.run:
        generate_histogram_plots(
            data=data,
            path=paths.Paths.HISTOGRAM_PLOTS_FOR_AGGREGATED_FEATURES_PATH.value,
            number_of_bins=config.generate_histogram_plots.number_of_bins,
        )
    if config.generate_custom_plots:
        custom_plots(
            data=data,
            path=paths.Paths.CUSTOM_PLOTS_FOR_AGGREGATED_FEATURES_PATH.value
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

def run_feature_engineering(data, config):
    if config.data.feature_engineering.run:
        data = feature_engineering(
            data=data
        )
        if config.export_data:
            export_data(
                data=data,
                path=paths.Paths.EXPORTED_DATA_FILE.value,
                path_suffix="stage_8_feature_engineered_data",
            )
        if config.generate_meta_data:
            generate_meta_data(
                data=data,
                path=paths.Paths.META_DATA_FILE.value,
                path_suffix="stage_8_feature_engineered_data",
            )
        if config.generate_time_series_plots:
            generate_time_series_plots(
                data=data,
                timestamp=config.data.timestamp,
                path=paths.Paths.TIME_SERIES_PLOT_FOR_FEATURE_ENGINEERED_DATA_PATH.value,
            )
        if config.generate_histogram_plots.run:
            generate_histogram_plots(
                data=data,
                path=paths.Paths.HISTOGRAM_PLOTS_FOR_FEATURE_ENGINEERED_DATA_PATH.value,
                number_of_bins=config.generate_histogram_plots.number_of_bins,
            )
        if config.generate_custom_plots:
            custom_plots(
                data=data,
                path=paths.Paths.CUSTOM_PLOTS_FOR_FEATURE_ENGINEERED_DATA_PATH.value
            )
    return data

def run_filter_shutdowns(data, shutdown_dates, config):
    if config.data.filter_shutdowns.run:
        if shutdown_dates is not None:
            data = data[~data['DATE'].isin(shutdown_dates)]
        if config.export_data:
            export_data(
                data=data,
                path=paths.Paths.EXPORTED_DATA_FILE.value,
                path_suffix="stage_10_filter_shutdowns",
            )
        if config.generate_meta_data:
            generate_meta_data(
                data=data,
                path=paths.Paths.META_DATA_FILE.value,
                path_suffix="stage_10_filter_shutdowns",
            )
        if config.generate_time_series_plots:
            generate_time_series_plots(
                data=data,
                timestamp=config.data.timestamp,
                path=paths.Paths.TIME_SERIES_PLOTS_FOR_FILTERING_SHUTDOWN_PATH.value,
            )
        if config.generate_histogram_plots.run:
            generate_histogram_plots(
                data=data,
                path=paths.Paths.HISTOGRAM_PLOTS_FOR_FILTERING_SHUTDOWN_PATH.value,
                number_of_bins=config.generate_histogram_plots.number_of_bins,
            )
        if config.generate_custom_plots:
            custom_plots(
                data=data,
                path=paths.Paths.CUSTOM_PLOTS_FOR_FILTERING_SHUTDOWN_PATH.value
            )
    return data

def generating_univariabe_feature_importance_for_iron_concentrate_perc_model(data, config):
    if config.iron_concentrate_perc.model.univariable_feature_importance.run:
        univariable_feature_importance = generate_univariable_feature_importance(
            data=data,
            target_feature=config.iron_concentrate_perc.model.target,
            method=config.iron_concentrate_perc.model.univariable_feature_importance.method,
            path=paths.Paths.IRON_CONCENTRATE_PERC_UNIVARIABLE_FEATURE_IMPORTANCE_PATH.value,
        )
        return univariable_feature_importance
    
def generating_univariabe_feature_importance_for_silica_concentrate_perc_model(data, config):
    if config.silica_concentrate_perc.model.univariable_feature_importance.run:
        univariable_feature_importance = generate_univariable_feature_importance(
            data=data,
            target_feature=config.silica_concentrate_perc.model.target,
            method=config.silica_concentrate_perc.model.univariable_feature_importance.method,
            path=paths.Paths.SILICA_CONCENTRATE_PERC_UNIVARIABLE_FEATURE_IMPORTANCE_PATH.value,
        )
        return univariable_feature_importance

def run_feature_selection_iron_concentrate_perc_model(data, config):
    if config.iron_concentrate_perc.model.feature_selection.run:
        (
            training_features,
            training_features_per_method,
            univariable_feature_importance_from_feature_selection,
            feature_importance_from_feature_selection,
        ) = feature_selection(
            data=data,
            target_feature=config.iron_concentrate_perc.model.target,
            filter_low_variance=config.iron_concentrate_perc.model.feature_selection.filter_low_variance.run,
            filter_univariable_feature_importance=config.iron_concentrate_perc.model.feature_selection.filter_univariable_feature_importance.run,
            filter_feature_importance=config.iron_concentrate_perc.model.feature_selection.filter_feature_importance.run,
            low_variance_threshold=config.iron_concentrate_perc.model.feature_selection.filter_low_variance.threshold,
            univariable_feature_importance_method=config.iron_concentrate_perc.model.univariable_feature_importance.method,
            univariable_feature_importance_threshold=config.iron_concentrate_perc.model.feature_selection.filter_univariable_feature_importance.threshold,
            feature_importance_model_choice=config.iron_concentrate_perc.model.model_choice,
            feature_importance_param_grid=config.iron_concentrate_perc.model.hyperparameters,
            feature_importance_threshold=config.iron_concentrate_perc.model.feature_selection.filter_feature_importance.threshold,
            training_features=config.iron_concentrate_perc.model.training_features,
        )
    else:
        training_features = config.iron_concentrate_perc.model.training_features
        training_features_per_method = None

    return training_features, training_features_per_method

def run_feature_selection_iron_concentrate_perc_feed_blend_model(data, config):
    if config.iron_concentrate_perc.model.feature_selection.run:
        (
            training_features,
            training_features_per_method,
            univariable_feature_importance_from_feature_selection,
            feature_importance_from_feature_selection,
        ) = feature_selection(
            data=data,
            target_feature=config.iron_concentrate_perc.model.target,
            filter_low_variance=config.iron_concentrate_perc.model.feature_selection.filter_low_variance.run,
            filter_univariable_feature_importance=config.iron_concentrate_perc.model.feature_selection.filter_univariable_feature_importance.run,
            filter_feature_importance=config.iron_concentrate_perc.model.feature_selection.filter_feature_importance.run,
            low_variance_threshold=config.iron_concentrate_perc.model.feature_selection.filter_low_variance.threshold,
            univariable_feature_importance_method=config.iron_concentrate_perc.model.univariable_feature_importance.method,
            univariable_feature_importance_threshold=config.iron_concentrate_perc.model.feature_selection.filter_univariable_feature_importance.threshold,
            feature_importance_model_choice=config.iron_concentrate_perc.model.model_choice,
            feature_importance_param_grid=config.iron_concentrate_perc.model.hyperparameters,
            feature_importance_threshold=config.iron_concentrate_perc.model.feature_selection.filter_feature_importance.threshold,
            training_features=config.iron_concentrate_perc.model.feed_blend_training_features,
        )
    else:
        training_features = config.iron_concentrate_perc.model.feed_blend_training_features
        training_features_per_method = None

    return training_features, training_features_per_method

def run_feature_selection_silica_concentrate_perc_model(data, config):
    if config.silica_concentrate_perc.model.feature_selection.run:
        (
            training_features,
            training_features_per_method,
            univariable_feature_importance_from_feature_selection,
            feature_importance_from_feature_selection,
        ) = feature_selection(
            data=data,
            target_feature=config.silica_concentrate_perc.model.target,
            filter_low_variance=config.silica_concentrate_perc.model.feature_selection.filter_low_variance.run,
            filter_univariable_feature_importance=config.silica_concentrate_perc.model.feature_selection.filter_univariable_feature_importance.run,
            filter_feature_importance=config.silica_concentrate_perc.model.feature_selection.filter_feature_importance.run,
            low_variance_threshold=config.silica_concentrate_perc.model.feature_selection.filter_low_variance.threshold,
            univariable_feature_importance_method=config.silica_concentrate_perc.model.univariable_feature_importance.method,
            univariable_feature_importance_threshold=config.silica_concentrate_perc.model.feature_selection.filter_univariable_feature_importance.threshold,
            feature_importance_model_choice=config.silica_concentrate_perc.model.model_choice,
            feature_importance_param_grid=config.silica_concentrate_perc.model.hyperparameters,
            feature_importance_threshold=config.silica_concentrate_perc.model.feature_selection.filter_feature_importance.threshold,
            training_features=config.silica_concentrate_perc.model.training_features,
        )
    else:
        training_features = config.silica_concentrate_perc.model.training_features
        training_features_per_method = None

    return training_features, training_features_per_method

def run_feature_selection_silica_concentrate_perc_feed_blend_model(data, config):
    if config.silica_concentrate_perc.model.feature_selection.run:
        (
            training_features,
            training_features_per_method,
            univariable_feature_importance_from_feature_selection,
            feature_importance_from_feature_selection,
        ) = feature_selection(
            data=data,
            target_feature=config.silica_concentrate_perc.model.target,
            filter_low_variance=config.silica_concentrate_perc.model.feature_selection.filter_low_variance.run,
            filter_univariable_feature_importance=config.silica_concentrate_perc.model.feature_selection.filter_univariable_feature_importance.run,
            filter_feature_importance=config.silica_concentrate_perc.model.feature_selection.filter_feature_importance.run,
            low_variance_threshold=config.silica_concentrate_perc.model.feature_selection.filter_low_variance.threshold,
            univariable_feature_importance_method=config.silica_concentrate_perc.model.univariable_feature_importance.method,
            univariable_feature_importance_threshold=config.silica_concentrate_perc.model.feature_selection.filter_univariable_feature_importance.threshold,
            feature_importance_model_choice=config.silica_concentrate_perc.model.model_choice,
            feature_importance_param_grid=config.silica_concentrate_perc.model.hyperparameters,
            feature_importance_threshold=config.silica_concentrate_perc.model.feature_selection.filter_feature_importance.threshold,
            training_features=config.silica_concentrate_perc.model.feed_blend_training_features,
        )
    else:
        training_features = config.silica_concentrate_perc.model.feed_blend_training_features
        training_features_per_method = None

    return training_features, training_features_per_method

def generating_correlation_matrix_iron_concentrate_perc_model(data, training_features_per_method, config):
    if config.iron_concentrate_perc.model.correlation_matrix.run:
        if training_features_per_method is None:
            training_features = config.iron_concentrate_perc.model.training_features + [config.iron_concentrate_perc.model.target]
        else:
            training_features = training_features_per_method[-1] + [config.iron_concentrate_perc.model.target]

        correlation_matrix = generate_correlation_matrix(
            data=data,
            features=training_features,
            method=config.iron_concentrate_perc.model.correlation_matrix.method,
            csv_path=paths.Paths.IRON_CONCENTRATE_PERC_CORRELATION_MATRIX_CSV_PATH.value,
            plotting_path=paths.Paths.IRON_CONCENTRATE_PERC_CORRELATION_MATRIX_PLOTTING_PATH.value,
        )
        return correlation_matrix
    
def generating_correlation_matrix_iron_concentrate_perc_feed_blend_model(data, training_features_per_method, config):
    if config.iron_concentrate_perc.model.correlation_matrix.run:
        if training_features_per_method is None:
            training_features = config.iron_concentrate_perc.model.feed_blend_training_features + [config.iron_concentrate_perc.model.target]
        else:
            training_features = training_features_per_method[-1] + [config.iron_concentrate_perc.model.target]

        correlation_matrix = generate_correlation_matrix(
            data=data,
            features=training_features,
            method=config.iron_concentrate_perc.model.correlation_matrix.method,
            csv_path=paths.Paths.IRON_CONCENTRATE_PERC_FEED_BLEND_CORRELATION_MATRIX_CSV_PATH.value,
            plotting_path=paths.Paths.IRON_CONCENTRATE_PERC_FEED_BLEND_CORRELATION_MATRIX_PLOTTING_PATH.value,
        )
        return correlation_matrix

def generating_correlation_matrix_silica_concentrate_perc_model(data, training_features_per_method, config):
    if config.silica_concentrate_perc.model.correlation_matrix.run:
        if training_features_per_method is None:
            training_features = config.silica_concentrate_perc.model.training_features + [config.silica_concentrate_perc.model.target]
        else:
            training_features = training_features_per_method[-1] + [config.silica_concentrate_perc.model.target]

        correlation_matrix = generate_correlation_matrix(
            data=data,
            features=training_features,
            method=config.silica_concentrate_perc.model.correlation_matrix.method,
            csv_path=paths.Paths.SILICA_CONCENTRATE_PERC_CORRELATION_MATRIX_CSV_PATH.value,
            plotting_path=paths.Paths.SILICA_CONCENTRATE_PERC_CORRELATION_MATRIX_PLOTTING_PATH.value,
        )
        return correlation_matrix
    
def generating_correlation_matrix_silica_concentrate_perc_feed_blend_model(data, training_features_per_method, config):
    if config.silica_concentrate_perc.model.correlation_matrix.run:
        if training_features_per_method is None:
            training_features = config.silica_concentrate_perc.model.feed_blend_training_features + [config.silica_concentrate_perc.model.target]
        else:
            training_features = training_features_per_method[-1] + [config.silica_concentrate_perc.model.target]

        correlation_matrix = generate_correlation_matrix(
            data=data,
            features=training_features,
            method=config.silica_concentrate_perc.model.correlation_matrix.method,
            csv_path=paths.Paths.SILICA_CONCENTRATE_PERC_FEED_BLEND_CORRELATION_MATRIX_CSV_PATH.value,
            plotting_path=paths.Paths.SILICA_CONCENTRATE_PERC_FEED_BLEND_CORRELATION_MATRIX_PLOTTING_PATH.value,
        )
        return correlation_matrix

def generating_model_iron_concentrate_perc_model(data, training_features, config):
    best_models, best_params, best_rmse, feature_importance = generate_model(
        data=data,
        target_feature=config.iron_concentrate_perc.model.target,
        training_features=training_features,
        model_choice=config.iron_concentrate_perc.model.model_choice,
        param_grid=config.iron_concentrate_perc.model.hyperparameters,
        metric=config.iron_concentrate_perc.model.metric,
        generate_feature_importance=config.iron_concentrate_perc.model.generate_feature_importance,
        feature_importance_path=paths.Paths.IRON_CONCENTRATE_PERC_FEATURE_IMPORTANCE_FILE.value,
        evaluation_results_path=paths.Paths.IRON_CONCENTRATE_PERC_MODEL_EVALUATION_RESULTS_FILE.value,
        random_state=config.iron_concentrate_perc.model.random_state,
        n_models=config.iron_concentrate_perc.model.number_of_models,
        path=paths.Paths.IRON_CONCENTRATE_PERC_MODEL_EVALUATION_SCATTER_PLOT.value
    )
    return best_models, best_params, best_rmse, feature_importance

def generating_model_iron_concentrate_perc_feed_blend_model(data, training_features, config):
    best_models, best_params, best_rmse, feature_importance = generate_model(
        data=data,
        target_feature=config.iron_concentrate_perc.model.target,
        training_features=training_features,
        model_choice=config.iron_concentrate_perc.model.model_choice,
        param_grid=config.iron_concentrate_perc.model.hyperparameters,
        metric=config.iron_concentrate_perc.model.metric,
        generate_feature_importance=config.iron_concentrate_perc.model.generate_feature_importance,
        feature_importance_path=paths.Paths.IRON_CONCENTRATE_PERC_FEED_BLEND_FEATURE_IMPORTANCE_FILE.value,
        evaluation_results_path=paths.Paths.IRON_CONCENTRATE_PERC_FEED_BLEND_MODEL_EVALUATION_RESULTS_FILE.value,
        random_state=config.iron_concentrate_perc.model.random_state,
        n_models=config.iron_concentrate_perc.model.number_of_models,
        path=paths.Paths.IRON_CONCENTRATE_PERC_FEED_BLEND_MODEL_EVALUATION_SCATTER_PLOT.value
    )
    return best_models, best_params, best_rmse, feature_importance

def generating_model_silica_concentrate_perc_model(data, training_features, config):
    best_models, best_params, best_rmse, feature_importance = generate_model(
        data=data,
        target_feature=config.silica_concentrate_perc.model.target,
        training_features=training_features,
        model_choice=config.silica_concentrate_perc.model.model_choice,
        param_grid=config.silica_concentrate_perc.model.hyperparameters,
        metric=config.silica_concentrate_perc.model.metric,
        generate_feature_importance=config.silica_concentrate_perc.model.generate_feature_importance,
        feature_importance_path=paths.Paths.SILICA_CONCENTRATE_PERC_FEATURE_IMPORTANCE_FILE.value,
        evaluation_results_path=paths.Paths.SILICA_CONCENTRATE_PERC_MODEL_EVALUATION_RESULTS_FILE.value,
        random_state=config.silica_concentrate_perc.model.random_state,
        n_models=config.silica_concentrate_perc.model.number_of_models,
        path=paths.Paths.SILICA_CONCENTRATE_PERC_MODEL_EVALUATION_SCATTER_PLOT.value
    )
    return best_models, best_params, best_rmse, feature_importance

def generating_model_silica_concentrate_perc_feed_blend_model(data, training_features, config):
    best_models, best_params, best_rmse, feature_importance = generate_model(
        data=data,
        target_feature=config.silica_concentrate_perc.model.target,
        training_features=training_features,
        model_choice=config.silica_concentrate_perc.model.model_choice,
        param_grid=config.silica_concentrate_perc.model.hyperparameters,
        metric=config.silica_concentrate_perc.model.metric,
        generate_feature_importance=config.silica_concentrate_perc.model.generate_feature_importance,
        feature_importance_path=paths.Paths.SILICA_CONCENTRATE_PERC_FEED_BLEND_FEATURE_IMPORTANCE_FILE.value,
        evaluation_results_path=paths.Paths.SILICA_CONCENTRATE_PERC_FEED_BLEND_MODEL_EVALUATION_RESULTS_FILE.value,
        random_state=config.silica_concentrate_perc.model.random_state,
        n_models=config.silica_concentrate_perc.model.number_of_models,
        path=paths.Paths.SILICA_CONCENTRATE_PERC_FEED_BLEND_MODEL_EVALUATION_SCATTER_PLOT.value
    )
    return best_models, best_params, best_rmse, feature_importance

def saving_models_iron_concentrate_perc_model(best_models, config):
    save_models(
        models=best_models,
        path=paths.Paths.IRON_CONCENTRATE_PERC_MODELS_FOLDER.value,
        path_suffix=config.iron_concentrate_perc.model.model_name,
    )

def saving_models_iron_concentrate_perc_feed_blend_model(best_models, config):
    save_models(
        models=best_models,
        path=paths.Paths.IRON_CONCENTRATE_PERC_FEED_BLEND_MODELS_FOLDER.value,
        path_suffix=config.iron_concentrate_perc.model.model_name,
    )

def saving_models_silica_concentrate_perc_model(best_models, config):
    save_models(
        models=best_models,
        path=paths.Paths.SILICA_CONCENTRATE_PERC_MODELS_FOLDER.value,
        path_suffix=config.silica_concentrate_perc.model.model_name,
    )

def saving_models_silica_concentrate_perc_feed_blend_model(best_models, config):
    save_models(
        models=best_models,
        path=paths.Paths.SILICA_CONCENTRATE_PERC_FEED_BLEND_MODELS_FOLDER.value,
        path_suffix=config.silica_concentrate_perc.model.model_name,
    )

def generating_partial_plots_iron_concentrate_perc_model(best_models, data, training_features, config):
    generate_partial_plots(
        model_choice=config.iron_concentrate_perc.model.model_choice,
        models=best_models,
        data=data[training_features],
        features=training_features,
        path=paths.Paths.IRON_CONCENTRATE_PERC_PARTIAL_PLOTS_PATH.value,
        plot_confidence_interval=config.iron_concentrate_perc.partial_plots.plot_confidence_interval,
        plot_feature_density=config.iron_concentrate_perc.partial_plots.plot_feature_density,
        plot_feature_density_as_histogram=config.iron_concentrate_perc.partial_plots.plot_feature_density_as_histogram,
        number_of_bins_in_histogram=config.iron_concentrate_perc.partial_plots.number_of_bins_in_histogram,
        grid_resolution=config.iron_concentrate_perc.partial_plots.grid_resolution,
    )

def generating_partial_plots_iron_concentrate_perc_feed_blend_model(best_models, data, training_features, config):
    generate_partial_plots(
        model_choice=config.iron_concentrate_perc.model.model_choice,
        models=best_models,
        data=data[training_features],
        features=training_features,
        path=paths.Paths.IRON_CONCENTRATE_PERC_FEED_BLEND_PARTIAL_PLOTS_PATH.value,
        plot_confidence_interval=config.iron_concentrate_perc.partial_plots.plot_confidence_interval,
        plot_feature_density=config.iron_concentrate_perc.partial_plots.plot_feature_density,
        plot_feature_density_as_histogram=config.iron_concentrate_perc.partial_plots.plot_feature_density_as_histogram,
        number_of_bins_in_histogram=config.iron_concentrate_perc.partial_plots.number_of_bins_in_histogram,
        grid_resolution=config.iron_concentrate_perc.partial_plots.grid_resolution,
    )

def generating_partial_plots_silica_concentrate_perc_model(best_models, data, training_features, config):
    generate_partial_plots(
        model_choice=config.silica_concentrate_perc.model.model_choice,
        models=best_models,
        data=data[training_features],
        features=training_features,
        path=paths.Paths.SILICA_CONCENTRATE_PERC_PARTIAL_PLOTS_PATH.value,
        plot_confidence_interval=config.silica_concentrate_perc.partial_plots.plot_confidence_interval,
        plot_feature_density=config.silica_concentrate_perc.partial_plots.plot_feature_density,
        plot_feature_density_as_histogram=config.silica_concentrate_perc.partial_plots.plot_feature_density_as_histogram,
        number_of_bins_in_histogram=config.silica_concentrate_perc.partial_plots.number_of_bins_in_histogram,
        grid_resolution=config.silica_concentrate_perc.partial_plots.grid_resolution,
    )

def generating_partial_plots_silica_concentrate_perc_feed_blend_model(best_models, data, training_features, config):
    generate_partial_plots(
        model_choice=config.silica_concentrate_perc.model.model_choice,
        models=best_models,
        data=data[training_features],
        features=training_features,
        path=paths.Paths.SILICA_CONCENTRATE_PERC_FEED_BLEND_PARTIAL_PLOTS_PATH.value,
        plot_confidence_interval=config.silica_concentrate_perc.partial_plots.plot_confidence_interval,
        plot_feature_density=config.silica_concentrate_perc.partial_plots.plot_feature_density,
        plot_feature_density_as_histogram=config.silica_concentrate_perc.partial_plots.plot_feature_density_as_histogram,
        number_of_bins_in_histogram=config.silica_concentrate_perc.partial_plots.number_of_bins_in_histogram,
        grid_resolution=config.silica_concentrate_perc.partial_plots.grid_resolution,
    )

def generating_clusters_feed_blend_model(data, config):
    clusters = run_clustering(
        data=data, 
        training_features=config.clustering.feed_blend_model.training_features,
        informational_features=config.clustering.feed_blend_model.informational_features,
        include_row_count_sum=config.clustering.feed_blend_model.include_row_count_sum,
        path=paths.Paths.FEED_BLEND_CLUSTERING_FILE.value,
        model_choice=config.clustering.feed_blend_model.model_choice,
        k_means_n_clusters=config.clustering.feed_blend_model.k_means_n_clusters,
        k_means_max_iter=config.clustering.feed_blend_model.k_means_max_iter,
        dbscan_eps=config.clustering.feed_blend_model.dbscan_eps,
        dbscan_min_samples=config.clustering.feed_blend_model.dbscan_min_samples,
        agglomerative_n_clusters=config.clustering.feed_blend_model.agglomerative_n_clusters,
        random_state=config.clustering.feed_blend_model.random_state
    )
    return clusters

def generating_simulations_iron_concentrate_perc_model(best_models, cluster_centers, config):
    simulation_results = generate_simulations(
        features=config.iron_concentrate_perc.model.training_features,
        feature_values_to_simulate=config.simulation.feed_blend_and_controllables_model.feature_values_to_simulate,
        model_choice=config.iron_concentrate_perc.model.model_choice,
        best_models=best_models,
        confidence_interval=config.simulation.feed_blend_and_controllables_model.confidence_interval,
        cluster_centers=cluster_centers,
        feed_blend_and_reagent_modelling=True,
        reagent_features=config.clustering.controllables_model.training_features
        #path=paths.Paths.FEED_BLEND_AND_CONTROLLABLES_IRON_CONCENTRATE_PERC_SIMULATIONS_FILE.value,
        #shapley_path=paths.Paths.SHAPLEY_RESULTS_FILE.value,
        #shapley_plots_path=paths.Paths.SHAPLEY_PLOTS_PATH.value,
    )
    return simulation_results

def generating_simulations_iron_concentrate_perc_feed_blend_model(best_models, cluster_centers, config):
    simulation_results = generate_simulations(
        features=config.iron_concentrate_perc.model.feed_blend_training_features,
        feature_values_to_simulate=config.simulation.feed_blend_model.feature_values_to_simulate,
        model_choice=config.iron_concentrate_perc.model.model_choice,
        best_models=best_models,
        confidence_interval=config.simulation.feed_blend_model.confidence_interval,
        cluster_centers=cluster_centers,
        informational_features=config.clustering.feed_blend_model.informational_features,
        feed_blend_and_reagent_modelling=False
        #path=paths.Paths.FEED_BLEND_IRON_CONCENTRATE_PERC_SIMULATIONS_FILE.value,
        #shapley_path=paths.Paths.SHAPLEY_RESULTS_FILE.value,
        #shapley_plots_path=paths.Paths.SHAPLEY_PLOTS_PATH.value,
    )
    return simulation_results

def generating_simulations_silica_concentrate_perc_model(best_models, cluster_centers, config):
    simulation_results = generate_simulations(
        features=config.silica_concentrate_perc.model.training_features,
        feature_values_to_simulate=config.simulation.feed_blend_and_controllables_model.feature_values_to_simulate,
        model_choice=config.silica_concentrate_perc.model.model_choice,
        best_models=best_models,
        confidence_interval=config.simulation.feed_blend_and_controllables_model.confidence_interval,
        cluster_centers=cluster_centers,
        feed_blend_and_reagent_modelling=True,
        reagent_features=config.clustering.controllables_model.training_features
        #path=paths.Paths.FEED_BLEND_AND_CONTROLLABLES_SILICA_CONCENTRATE_PERC_SIMULATIONS_FILE.value,
        #shapley_path=paths.Paths.SHAPLEY_RESULTS_FILE.value,
        #shapley_plots_path=paths.Paths.SHAPLEY_PLOTS_PATH.value,
    )
    return simulation_results

def generating_simulations_silica_concentrate_perc_feed_blend_model(best_models, cluster_centers, config):
    simulation_results = generate_simulations(
        features=config.silica_concentrate_perc.model.feed_blend_training_features,
        feature_values_to_simulate=config.simulation.feed_blend_model.feature_values_to_simulate,
        model_choice=config.silica_concentrate_perc.model.model_choice,
        best_models=best_models,
        confidence_interval=config.simulation.feed_blend_model.confidence_interval,
        cluster_centers=cluster_centers,
        informational_features=config.clustering.feed_blend_model.informational_features,
        feed_blend_and_reagent_modelling=False
        #path=paths.Paths.FEED_BLEND_SILICA_CONCENTRATE_PERC_SIMULATIONS_FILE.value,
        #shapley_path=paths.Paths.SHAPLEY_RESULTS_FILE.value,
        #shapley_plots_path=paths.Paths.SHAPLEY_PLOTS_PATH.value,
    )
    return simulation_results

def combine_feed_blend_and_controllables_simulations(iron_concentrate_perc_simulation_results,
                                                     silica_concentrate_perc_simulation_results,
                                                     config):

    # Combining the simulations
    iron_concentrate_perc_simulation_results = iron_concentrate_perc_simulation_results.rename(columns={'mean_simulated_predictions': 'IRON_CONCENTRATE_PERC_mean_simulated_predictions'})
    silica_concentrate_perc_simulation_results = silica_concentrate_perc_simulation_results.rename(columns={'mean_simulated_predictions': 'SILICA_CONCENTRATE_PERC_mean_simulated_predictions'})

    iron_concentrate_perc_simulation_results['SILICA_CONCENTRATE_PERC_mean_simulated_predictions'] = silica_concentrate_perc_simulation_results['SILICA_CONCENTRATE_PERC_mean_simulated_predictions']

    # Outputting to a csv file
    iron_concentrate_perc_simulation_results.to_csv(paths.Paths.FEED_BLEND_AND_CONTROLLABLES_SIMULATIONS_FILE.value, index=False)

    return iron_concentrate_perc_simulation_results

def combine_feed_blend_simulations(iron_concentrate_perc_feed_blend_simulation_results,
                                   silica_concentrate_perc_feed_blend_simulation_results,
                                   config):

    # Identiying the historical predictions from each of the feed blend simulations
    iron_concentrate_perc_feed_blend_simulation_results = iron_concentrate_perc_feed_blend_simulation_results.rename(columns={'mean_historical_predictions': 'IRON_CONCENTRATE_PERC_mean_historical_predictions'})
    silica_concentrate_perc_feed_blend_simulation_results = silica_concentrate_perc_feed_blend_simulation_results.rename(columns={'mean_historical_predictions': 'SILICA_CONCENTRATE_PERC_mean_historical_predictions'})

    # Combining the simulations
    iron_concentrate_perc_feed_blend_simulation_results['SILICA_CONCENTRATE_PERC_mean_historical_predictions'] = silica_concentrate_perc_feed_blend_simulation_results['SILICA_CONCENTRATE_PERC_mean_historical_predictions']

    # Outputting to a csv file
    iron_concentrate_perc_feed_blend_simulation_results.to_csv(paths.Paths.FEED_BLEND_SIMULATIONS_FILE.value, index=False)

    return iron_concentrate_perc_feed_blend_simulation_results

def generating_clusters_controllables_model(data, config):
    clusters = run_clustering(
        data=data,
        training_features=config.clustering.controllables_model.training_features,  
        informational_features=config.clustering.controllables_model.informational_features,
        include_row_count_sum=config.clustering.controllables_model.include_row_count_sum,
        path=paths.Paths.CONTROLLABLES_CLUSTERING_FILE.value,
        model_choice=config.clustering.controllables_model.model_choice,
        k_means_n_clusters=config.clustering.controllables_model.k_means_n_clusters,
        k_means_max_iter=config.clustering.controllables_model.k_means_max_iter,
        dbscan_eps=config.clustering.controllables_model.dbscan_eps,
        dbscan_min_samples=config.clustering.controllables_model.dbscan_min_samples,
        agglomerative_n_clusters=config.clustering.controllables_model.agglomerative_n_clusters,
        random_state=config.clustering.controllables_model.random_state
    )
    return clusters

def generate_clusters_combinations(feed_blend_clusters, reagent_clusters, config):
    cluster_combination_centers = create_cluster_combinations(
        feed_blend_clusters = feed_blend_clusters,
        reagent_clusters = reagent_clusters,
        feed_blend_training_features = config.clustering.feed_blend_model.training_features,
        reagent_training_features = config.clustering.controllables_model.training_features,
        path = paths.Paths.COMBINED_CLUSTERING_FILE.value
    )

    return cluster_combination_centers

def override_values_in_clusters(clusters):
    
    # No overrides are currently occuring

    return clusters

def optimise_clusters(cluster_combination_centers, feed_blend_simulations, controllables_clusters, config):

    optimal_clusters = create_optimised_clusters(
        cluster_combination_centers=cluster_combination_centers,
        feed_blend_simulations=feed_blend_simulations,
        controllables_clusters=controllables_clusters,
        path=paths.Paths.OPTIMISED_CLUSTERS_FILE.value,
        feature_to_optimize=config.optimisation.feature_to_optimise,
        optimisation_direction=config.optimisation.direction_to_optimise,
        controllable_features=config.clustering.controllables_model.training_features,
        constraint_features=config.optimisation.constraints.features,
        constraint_limits_per_feature=config.optimisation.constraints.limits
    )

    return optimal_clusters

def identify_shutdown_times(data):
    
    # Currently no shutdown periods have been identified
    shutdown_dates = None

    return shutdown_dates