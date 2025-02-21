import config.paths as paths
from shared.model.classes.correlation_matrix_processor import CorrelationMatrixProcessor
from dataclasses import dataclass


@dataclass
class CorrelationMatrixGenerator:
    model_config: dict

    def run_for_iron_concentrate_perc(self, data, training_features_per_method):
        if self.model_config.iron_concentrate_perc.model.correlation_matrix.run:
            if training_features_per_method is None:
                training_features = (
                    self.model_config.iron_concentrate_perc.model.training_features
                    + [self.model_config.iron_concentrate_perc.model.target]
                )
            else:
                training_features = training_features_per_method[-1] + [
                    self.model_config.iron_concentrate_perc.model.target
                ]

            correlation_matrix_processor = CorrelationMatrixProcessor(
                data=data,
                features=training_features,
                method=self.model_config.iron_concentrate_perc.model.correlation_matrix.method,
                csv_path=paths.Paths.IRON_CONCENTRATE_PERC_CORRELATION_MATRIX_CSV_PATH.value,
                plotting_path=paths.Paths.IRON_CONCENTRATE_PERC_CORRELATION_MATRIX_PLOTTING_PATH.value
            )
            correlation_matrix = correlation_matrix_processor.run()
            return correlation_matrix

    def run_for_iron_concentrate_perc_feed_blend(
        self, data, training_features_per_method
    ):
        if self.model_config.iron_concentrate_perc.model.correlation_matrix.run:
            if training_features_per_method is None:
                training_features = (
                    self.model_config.iron_concentrate_perc.model.feed_blend_training_features
                    + [self.model_config.iron_concentrate_perc.model.target]
                )
            else:
                training_features = training_features_per_method[-1] + [
                    self.model_config.iron_concentrate_perc.model.target
                ]

            correlation_matrix_processor = CorrelationMatrixProcessor(
                data=data,
                features=training_features,
                method=self.model_config.iron_concentrate_perc.model.correlation_matrix.method,
                csv_path=paths.Paths.IRON_CONCENTRATE_PERC_FEED_BLEND_CORRELATION_MATRIX_CSV_PATH.value,
                plotting_path=paths.Paths.IRON_CONCENTRATE_PERC_FEED_BLEND_CORRELATION_MATRIX_PLOTTING_PATH.value
            )
            correlation_matrix = correlation_matrix_processor.run()
            return correlation_matrix

    def run_for_silica_concentrate_perc(self, data, training_features_per_method):
        if self.model_config.silica_concentrate_perc.model.correlation_matrix.run:
            if training_features_per_method is None:
                training_features = (
                    self.model_config.silica_concentrate_perc.model.training_features
                    + [self.model_config.silica_concentrate_perc.model.target]
                )
            else:
                training_features = training_features_per_method[-1] + [
                    self.model_config.silica_concentrate_perc.model.target
                ]

            correlation_matrix_processor = CorrelationMatrixProcessor(
                data=data,
                features=training_features,
                method=self.model_config.silica_concentrate_perc.model.correlation_matrix.method,
                csv_path=paths.Paths.SILICA_CONCENTRATE_PERC_CORRELATION_MATRIX_CSV_PATH.value,
                plotting_path=paths.Paths.SILICA_CONCENTRATE_PERC_CORRELATION_MATRIX_PLOTTING_PATH.value
            )
            correlation_matrix = correlation_matrix_processor.run()
            return correlation_matrix

    def run_for_silica_concentrate_perc_feed_blend(
        self, data, training_features_per_method
    ):
        if self.model_config.silica_concentrate_perc.model.correlation_matrix.run:
            if training_features_per_method is None:
                training_features = (
                    self.model_config.silica_concentrate_perc.model.feed_blend_training_features
                    + [self.model_config.silica_concentrate_perc.model.target]
                )
            else:
                training_features = training_features_per_method[-1] + [
                    self.model_config.silica_concentrate_perc.model.target
                ]
            
            correlation_matrix_processor = CorrelationMatrixProcessor(
                data=data,
                features=training_features,
                method=self.model_config.silica_concentrate_perc.model.correlation_matrix.method,
                csv_path=paths.Paths.SILICA_CONCENTRATE_PERC_FEED_BLEND_CORRELATION_MATRIX_CSV_PATH.value,
                plotting_path=paths.Paths.SILICA_CONCENTRATE_PERC_FEED_BLEND_CORRELATION_MATRIX_PLOTTING_PATH.value
            )
            correlation_matrix = correlation_matrix_processor.run()
            return correlation_matrix