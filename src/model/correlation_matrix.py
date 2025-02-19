import config.paths as paths
from shared.model.generate_correlation_matrix import generate_correlation_matrix

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
