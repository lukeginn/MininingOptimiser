import config.paths as paths
from shared.model.generate_correlation_matrix import generate_correlation_matrix

class CorrelationMatrixGenerator:
    def __init__(self, config):
        self.config = config

    def run_for_iron_concentrate_perc(self, data, training_features_per_method):
        if self.config.iron_concentrate_perc.model.correlation_matrix.run:
            if training_features_per_method is None:
                training_features = self.config.iron_concentrate_perc.model.training_features + [self.config.iron_concentrate_perc.model.target]
            else:
                training_features = training_features_per_method[-1] + [self.config.iron_concentrate_perc.model.target]

            correlation_matrix = generate_correlation_matrix(
                data=data,
                features=training_features,
                method=self.config.iron_concentrate_perc.model.correlation_matrix.method,
                csv_path=paths.Paths.IRON_CONCENTRATE_PERC_CORRELATION_MATRIX_CSV_PATH.value,
                plotting_path=paths.Paths.IRON_CONCENTRATE_PERC_CORRELATION_MATRIX_PLOTTING_PATH.value,
            )
            return correlation_matrix

    def run_for_iron_concentrate_perc_feed_blend(self, data, training_features_per_method):
        if self.config.iron_concentrate_perc.model.correlation_matrix.run:
            if training_features_per_method is None:
                training_features = self.config.iron_concentrate_perc.model.feed_blend_training_features + [self.config.iron_concentrate_perc.model.target]
            else:
                training_features = training_features_per_method[-1] + [self.config.iron_concentrate_perc.model.target]

            correlation_matrix = generate_correlation_matrix(
                data=data,
                features=training_features,
                method=self.config.iron_concentrate_perc.model.correlation_matrix.method,
                csv_path=paths.Paths.IRON_CONCENTRATE_PERC_FEED_BLEND_CORRELATION_MATRIX_CSV_PATH.value,
                plotting_path=paths.Paths.IRON_CONCENTRATE_PERC_FEED_BLEND_CORRELATION_MATRIX_PLOTTING_PATH.value,
            )
            return correlation_matrix

    def run_for_silica_concentrate_perc(self, data, training_features_per_method):
        if self.config.silica_concentrate_perc.model.correlation_matrix.run:
            if training_features_per_method is None:
                training_features = self.config.silica_concentrate_perc.model.training_features + [self.config.silica_concentrate_perc.model.target]
            else:
                training_features = training_features_per_method[-1] + [self.config.silica_concentrate_perc.model.target]

            correlation_matrix = generate_correlation_matrix(
                data=data,
                features=training_features,
                method=self.config.silica_concentrate_perc.model.correlation_matrix.method,
                csv_path=paths.Paths.SILICA_CONCENTRATE_PERC_CORRELATION_MATRIX_CSV_PATH.value,
                plotting_path=paths.Paths.SILICA_CONCENTRATE_PERC_CORRELATION_MATRIX_PLOTTING_PATH.value,
            )
            return correlation_matrix

    def run_for_silica_concentrate_perc_feed_blend(self, data, training_features_per_method):
        if self.config.silica_concentrate_perc.model.correlation_matrix.run:
            if training_features_per_method is None:
                training_features = self.config.silica_concentrate_perc.model.feed_blend_training_features + [self.config.silica_concentrate_perc.model.target]
            else:
                training_features = training_features_per_method[-1] + [self.config.silica_concentrate_perc.model.target]

            correlation_matrix = generate_correlation_matrix(
                data=data,
                features=training_features,
                method=self.config.silica_concentrate_perc.model.correlation_matrix.method,
                csv_path=paths.Paths.SILICA_CONCENTRATE_PERC_FEED_BLEND_CORRELATION_MATRIX_CSV_PATH.value,
                plotting_path=paths.Paths.SILICA_CONCENTRATE_PERC_FEED_BLEND_CORRELATION_MATRIX_PLOTTING_PATH.value,
            )
            return correlation_matrix
