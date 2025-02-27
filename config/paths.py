from pathlib import Path
from enum import Enum


class Paths(Enum):
    BASE_PATH = Path().resolve()

    CONFIG_PATH = BASE_PATH / "config"
    GENERAL_CONFIG_FILE_PATH = CONFIG_PATH / "general.yaml"
    DATA_CONFIG_FILE_PATH = CONFIG_PATH / "data.yaml"
    MODEL_CONFIG_FILE_PATH = CONFIG_PATH / "model.yaml"
    CLUSTERING_CONFIG_FILE_PATH = CONFIG_PATH / "clustering.yaml"
    SIMULATION_CONFIG_FILE_PATH = CONFIG_PATH / "simulation.yaml"
    OPTIMISATION_CONFIG_FILE_PATH = CONFIG_PATH / "optimisation.yaml"

    INPUTS_PATH = BASE_PATH / "inputs"
    DATA_INPUTS_PATH = INPUTS_PATH / "data"
    DATA_FILE_1 = DATA_INPUTS_PATH / "MiningProcess_Flotation_Plant_Database.csv"

    CACHED_INPUTS_PATH = INPUTS_PATH / "cached_inputs"

    OUTPUTS_PATH = BASE_PATH / "outputs"
    META_DATA_PATH = OUTPUTS_PATH / "meta_data"
    META_DATA_FILE = META_DATA_PATH / "meta_data.csv"
    EXPORTED_DATA_PATH = OUTPUTS_PATH / "exported_data"
    EXPORTED_DATA_FILE = EXPORTED_DATA_PATH / "data.csv"

    TIME_SERIES_PLOTS_PATH = OUTPUTS_PATH / "time_series_plots"
    TIME_SERIES_PLOTS_FOR_RAW_DATA_PATH = TIME_SERIES_PLOTS_PATH / "stage_1_raw_data"
    TIME_SERIES_PLOTS_FOR_MISSING_DATA_IDENTIFIED_PATH = (
        TIME_SERIES_PLOTS_PATH / "stage_2_missing_data_identified"
    )
    TIME_SERIES_PLOTS_FOR_OUTLIERS_IDENTIFIED_PATH = (
        TIME_SERIES_PLOTS_PATH / "stage_3_outliers_identified"
    )
    TIME_SERIES_PLOTS_FOR_SHUTDOWNS_IDENTIFIED_PATH = (
        TIME_SERIES_PLOTS_PATH / "stage_4_shutdowns_identified"
    )
    TIME_SERIES_PLOTS_FOR_MISSING_DATA_CORRECTED_PATH = (
        TIME_SERIES_PLOTS_PATH / "stage_5_missing_data_corrected"
    )
    TIME_SERIES_PLOTS_FOR_LAGGED_FEATURES_PATH = (
        TIME_SERIES_PLOTS_PATH / "stage_6_lagged_features"
    )
    TIME_SERIES_PLOTS_FOR_AGGREGATED_FEATURES_PATH = (
        TIME_SERIES_PLOTS_PATH / "stage_7_rolling_aggregated_features"
    )
    TIME_SERIES_PLOT_FOR_FEATURE_ENGINEERED_DATA_PATH = (
        TIME_SERIES_PLOTS_PATH / "stage_8_feature_engineered_data"
    )
    TIME_SERIES_PLOTS_FOR_MISSING_DATA_CORRECTED_POST_AGGREGATION_PATH = (
        TIME_SERIES_PLOTS_PATH / "stage_9_missing_data_corrected_post_aggregation"
    )
    TIME_SERIES_PLOTS_FOR_FILTERING_SHUTDOWN_PATH = (
        TIME_SERIES_PLOTS_PATH / "stage_10_filtering_shutdowns"
    )
    TIME_SERIES_PLOTS_FOR_OPTIMISED_DATA_PATH = (
        TIME_SERIES_PLOTS_PATH / "stage_11_optimised_data"
    )

    HISTOGRAM_PLOTS_PATH = OUTPUTS_PATH / "histogram_plots"
    HISTOGRAM_PLOTS_FOR_RAW_DATA_PATH = HISTOGRAM_PLOTS_PATH / "stage_1_raw_data"
    HISTOGRAM_PLOTS_FOR_MISSING_DATA_IDENTIFIED_PATH = (
        HISTOGRAM_PLOTS_PATH / "stage_2_missing_data_identified"
    )
    HISTOGRAM_PLOTS_FOR_OUTLIERS_IDENTIFIED_PATH = (
        HISTOGRAM_PLOTS_PATH / "stage_3_outliers_identified"
    )
    HISTOGRAM_PLOTS_FOR_SHUTDOWNS_IDENTIFIED_PATH = (
        HISTOGRAM_PLOTS_PATH / "stage_4_shutdowns_identified"
    )
    HISTOGRAM_PLOTS_FOR_MISSING_DATA_CORRECTED_PATH = (
        HISTOGRAM_PLOTS_PATH / "stage_5_missing_data_corrected"
    )
    HISTOGRAM_PLOTS_FOR_LAGGED_FEATURES_PATH = (
        HISTOGRAM_PLOTS_PATH / "stage_6_lagged_features"
    )
    HISTOGRAM_PLOTS_FOR_AGGREGATED_FEATURES_PATH = (
        HISTOGRAM_PLOTS_PATH / "stage_7_rolling_aggregated_features"
    )
    HISTOGRAM_PLOTS_FOR_FEATURE_ENGINEERED_DATA_PATH = (
        HISTOGRAM_PLOTS_PATH / "stage_8_feature_engineered_data"
    )
    HISTOGRAM_PLOTS_FOR_MISSING_DATA_CORRECTED_POST_AGGREGATION_PATH = (
        HISTOGRAM_PLOTS_PATH / "stage_9_missing_data_corrected_post_aggregation"
    )
    HISTOGRAM_PLOTS_FOR_FILTERING_SHUTDOWN_PATH = (
        HISTOGRAM_PLOTS_PATH / "stage_10_filtering_shutdowns"
    )
    HISTOGRAM_PLOTS_FOR_OPTIMISED_DATA_PATH = (
        HISTOGRAM_PLOTS_PATH / "stage_11_optimised_data"
    )

    CUSTOM_PLOTS_PATH = OUTPUTS_PATH / "custom_plots"
    CUSTOM_PLOTS_FOR_RAW_DATA_PATH = CUSTOM_PLOTS_PATH / "stage_1_raw_data"
    CUSTOM_PLOTS_FOR_MISSING_DATA_IDENTIFIED_PATH = (
        CUSTOM_PLOTS_PATH / "stage_2_missing_data_identified"
    )
    CUSTOM_PLOTS_FOR_OUTLIERS_IDENTIFIED_PATH = (
        CUSTOM_PLOTS_PATH / "stage_3_outliers_identified"
    )
    CUSTOM_PLOTS_FOR_MISSING_DATA_CORRECTED_PATH = (
        CUSTOM_PLOTS_PATH / "stage_4_missing_data_corrected"
    )
    CUSTOM_PLOTS_FOR_SHUTDOWNS_IDENTIFIED_PATH = (
        CUSTOM_PLOTS_PATH / "stage_5_shutdowns_identified"
    )
    CUSTOM_PLOTS_FOR_LAGGED_FEATURES_PATH = (
        CUSTOM_PLOTS_PATH / "stage_6_lagged_features"
    )
    CUSTOM_PLOTS_FOR_AGGREGATED_FEATURES_PATH = (
        CUSTOM_PLOTS_PATH / "stage_7_rolling_aggregated_features"
    )
    CUSTOM_PLOTS_FOR_FEATURE_ENGINEERED_DATA_PATH = (
        CUSTOM_PLOTS_PATH / "stage_8_feature_engineered_data"
    )
    CUSTOM_PLOTS_FOR_MISSING_DATA_CORRECTED_POST_AGGREGATION_PATH = (
        CUSTOM_PLOTS_PATH / "stage_9_missing_data_corrected_post_aggregation"
    )
    CUSTOM_PLOTS_FOR_FILTERING_SHUTDOWN_PATH = (
        CUSTOM_PLOTS_PATH / "stage_10_filtering_shutdowns"
    )
    CUSTOM_PLOTS_FOR_OPTIMISED_DATA_PATH = CUSTOM_PLOTS_PATH / "stage_11_optimised_data"

    IRON_CONCENTRATE_PERC_MODELS_PATH = OUTPUTS_PATH / "iron_concentrate_perc_model"
    IRON_CONCENTRATE_PERC_CORRELATION_MATRIX_CSV_PATH = (
        IRON_CONCENTRATE_PERC_MODELS_PATH / "correlation_matrix.csv"
    )
    IRON_CONCENTRATE_PERC_CORRELATION_MATRIX_PLOTTING_PATH = (
        IRON_CONCENTRATE_PERC_MODELS_PATH / "correlation_matrix.png"
    )
    IRON_CONCENTRATE_PERC_MODELS_FOLDER = IRON_CONCENTRATE_PERC_MODELS_PATH / "models"
    IRON_CONCENTRATE_PERC_UNIVARIABLE_FEATURE_IMPORTANCE_PATH = (
        IRON_CONCENTRATE_PERC_MODELS_PATH / "univariable_feature_importance.csv"
    )
    IRON_CONCENTRATE_PERC_FEATURE_IMPORTANCE_FILE = (
        IRON_CONCENTRATE_PERC_MODELS_PATH / "feature_importance.csv"
    )
    IRON_CONCENTRATE_PERC_MODEL_EVALUATION_RESULTS_FILE = (
        IRON_CONCENTRATE_PERC_MODELS_PATH / "model_evaluation_results.csv"
    )
    IRON_CONCENTRATE_PERC_MODEL_EVALUATION_SCATTER_PLOT = (
        IRON_CONCENTRATE_PERC_MODELS_PATH / "model_evaluation_scatter_plot.png"
    )
    IRON_CONCENTRATE_PERC_PARTIAL_PLOTS_PATH = (
        IRON_CONCENTRATE_PERC_MODELS_PATH / "partial_plots"
    )

    IRON_CONCENTRATE_PERC_FEED_BLEND_MODELS_PATH = (
        OUTPUTS_PATH / "iron_concentrate_perc_feed_blend_model"
    )
    IRON_CONCENTRATE_PERC_FEED_BLEND_CORRELATION_MATRIX_CSV_PATH = (
        IRON_CONCENTRATE_PERC_FEED_BLEND_MODELS_PATH / "correlation_matrix.csv"
    )
    IRON_CONCENTRATE_PERC_FEED_BLEND_CORRELATION_MATRIX_PLOTTING_PATH = (
        IRON_CONCENTRATE_PERC_FEED_BLEND_MODELS_PATH / "correlation_matrix.png"
    )
    IRON_CONCENTRATE_PERC_FEED_BLEND_MODELS_FOLDER = (
        IRON_CONCENTRATE_PERC_FEED_BLEND_MODELS_PATH / "models"
    )
    IRON_CONCENTRATE_PERC_FEED_BLEND_UNIVARIABLE_FEATURE_IMPORTANCE_PATH = (
        IRON_CONCENTRATE_PERC_FEED_BLEND_MODELS_PATH
        / "univariable_feature_importance.csv"
    )
    IRON_CONCENTRATE_PERC_FEED_BLEND_FEATURE_IMPORTANCE_FILE = (
        IRON_CONCENTRATE_PERC_FEED_BLEND_MODELS_PATH / "feature_importance.csv"
    )
    IRON_CONCENTRATE_PERC_FEED_BLEND_MODEL_EVALUATION_RESULTS_FILE = (
        IRON_CONCENTRATE_PERC_FEED_BLEND_MODELS_PATH / "model_evaluation_results.csv"
    )
    IRON_CONCENTRATE_PERC_FEED_BLEND_MODEL_EVALUATION_SCATTER_PLOT = (
        IRON_CONCENTRATE_PERC_FEED_BLEND_MODELS_PATH
        / "model_evaluation_scatter_plot.png"
    )
    IRON_CONCENTRATE_PERC_FEED_BLEND_PARTIAL_PLOTS_PATH = (
        IRON_CONCENTRATE_PERC_FEED_BLEND_MODELS_PATH / "partial_plots"
    )

    SILICA_CONCENTRATE_PERC_MODELS_PATH = OUTPUTS_PATH / "silica_concentrate_perc_model"
    SILICA_CONCENTRATE_PERC_CORRELATION_MATRIX_CSV_PATH = (
        SILICA_CONCENTRATE_PERC_MODELS_PATH / "correlation_matrix.csv"
    )
    SILICA_CONCENTRATE_PERC_CORRELATION_MATRIX_PLOTTING_PATH = (
        SILICA_CONCENTRATE_PERC_MODELS_PATH / "correlation_matrix.png"
    )
    SILICA_CONCENTRATE_PERC_MODELS_FOLDER = (
        SILICA_CONCENTRATE_PERC_MODELS_PATH / "models"
    )
    SILICA_CONCENTRATE_PERC_UNIVARIABLE_FEATURE_IMPORTANCE_PATH = (
        SILICA_CONCENTRATE_PERC_MODELS_PATH / "univariable_feature_importance.csv"
    )
    SILICA_CONCENTRATE_PERC_FEATURE_IMPORTANCE_FILE = (
        SILICA_CONCENTRATE_PERC_MODELS_PATH / "feature_importance.csv"
    )
    SILICA_CONCENTRATE_PERC_MODEL_EVALUATION_RESULTS_FILE = (
        SILICA_CONCENTRATE_PERC_MODELS_PATH / "model_evaluation_results.csv"
    )
    SILICA_CONCENTRATE_PERC_MODEL_EVALUATION_SCATTER_PLOT = (
        SILICA_CONCENTRATE_PERC_MODELS_PATH / "model_evaluation_scatter_plot.png"
    )
    SILICA_CONCENTRATE_PERC_PARTIAL_PLOTS_PATH = (
        SILICA_CONCENTRATE_PERC_MODELS_PATH / "partial_plots"
    )

    SILICA_CONCENTRATE_PERC_FEED_BLEND_MODELS_PATH = (
        OUTPUTS_PATH / "silica_concentrate_perc_feed_blend_model"
    )
    SILICA_CONCENTRATE_PERC_FEED_BLEND_CORRELATION_MATRIX_CSV_PATH = (
        SILICA_CONCENTRATE_PERC_FEED_BLEND_MODELS_PATH / "correlation_matrix.csv"
    )
    SILICA_CONCENTRATE_PERC_FEED_BLEND_CORRELATION_MATRIX_PLOTTING_PATH = (
        SILICA_CONCENTRATE_PERC_FEED_BLEND_MODELS_PATH / "correlation_matrix.png"
    )
    SILICA_CONCENTRATE_PERC_FEED_BLEND_MODELS_FOLDER = (
        SILICA_CONCENTRATE_PERC_FEED_BLEND_MODELS_PATH / "models"
    )
    SILICA_CONCENTRATE_PERC_FEED_BLEND_UNIVARIABLE_FEATURE_IMPORTANCE_PATH = (
        SILICA_CONCENTRATE_PERC_FEED_BLEND_MODELS_PATH
        / "univariable_feature_importance.csv"
    )
    SILICA_CONCENTRATE_PERC_FEED_BLEND_FEATURE_IMPORTANCE_FILE = (
        SILICA_CONCENTRATE_PERC_FEED_BLEND_MODELS_PATH / "feature_importance.csv"
    )
    SILICA_CONCENTRATE_PERC_FEED_BLEND_MODEL_EVALUATION_RESULTS_FILE = (
        SILICA_CONCENTRATE_PERC_FEED_BLEND_MODELS_PATH / "model_evaluation_results.csv"
    )
    SILICA_CONCENTRATE_PERC_FEED_BLEND_MODEL_EVALUATION_SCATTER_PLOT = (
        SILICA_CONCENTRATE_PERC_FEED_BLEND_MODELS_PATH
        / "model_evaluation_scatter_plot.png"
    )
    SILICA_CONCENTRATE_PERC_FEED_BLEND_PARTIAL_PLOTS_PATH = (
        SILICA_CONCENTRATE_PERC_FEED_BLEND_MODELS_PATH / "partial_plots"
    )

    CLUSTERS_AND_SIMULATIONS_PATH = OUTPUTS_PATH / "clusters_and_simulations"

    CLUSTERS_PATH = CLUSTERS_AND_SIMULATIONS_PATH / "clusters"
    FEED_BLEND_CLUSTERING_FILE = CLUSTERS_PATH / "feed_blend_clusters.csv"
    CONTROLLABLES_CLUSTERING_FILE = CLUSTERS_PATH / "controllables_clusters.csv"
    COMBINED_CLUSTERING_FILE = (
        CLUSTERS_PATH / "feed_blend_and_controllables_combined_clusters.csv"
    )

    SIMULATIONS_PATH = CLUSTERS_AND_SIMULATIONS_PATH / "simulations"
    FEED_BLEND_SIMULATIONS_FILE = SIMULATIONS_PATH / "feed_blend_simulations.csv"
    FEED_BLEND_AND_CONTROLLABLES_SIMULATIONS_FILE = (
        SIMULATIONS_PATH / "feed_blend_and_controllables_simulations.csv"
    )
    OPTIMISED_CLUSTERS_FILE = SIMULATIONS_PATH / "optimised_clusters.csv"


def create_directories():
    for path in Paths:
        if path.value.suffix == "":  # Check if it's a directory (no suffix)
            path.value.mkdir(parents=True, exist_ok=True)
