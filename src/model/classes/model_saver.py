import config.paths as paths
from dataclasses import dataclass
from typing import Any, Dict
from shared.model.model_exporter import ModelExporter

@dataclass
class ModelSaver:
    model_config: Dict[str, Any]

    def run_for_iron_concentrate_perc(self, models: Dict[str, Any]):
        model_exporter = ModelExporter()
        model_exporter.save(
            models=models,
            path=paths.Paths.IRON_CONCENTRATE_PERC_MODELS_FOLDER.value,
            path_suffix=self.model_config["iron_concentrate_perc"]["model"]["model_name"],
        )

    def run_for_iron_concentrate_perc_feed_blend(self, models: Dict[str, Any]):
        model_exporter = ModelExporter()
        model_exporter.save(
            path=paths.Paths.IRON_CONCENTRATE_PERC_FEED_BLEND_MODELS_FOLDER.value,
            models=models,
            path_suffix=self.model_config["iron_concentrate_perc"]["model"]["model_name"],
        )

    def run_for_silica_concentrate_perc(self, models: Dict[str, Any]):
        model_exporter = ModelExporter()
        model_exporter.save(
            path=paths.Paths.SILICA_CONCENTRATE_PERC_MODELS_FOLDER.value,
            models=models,
            path_suffix=self.model_config["silica_concentrate_perc"]["model"]["model_name"],
        )

    def run_for_silica_concentrate_perc_feed_blend(self, models: Dict[str, Any]):
        model_exporter = ModelExporter()
        model_exporter.save(
            path=paths.Paths.SILICA_CONCENTRATE_PERC_FEED_BLEND_MODELS_FOLDER.value,
            models=models,
            path_suffix=self.model_config["silica_concentrate_perc"]["model"]["model_name"],
        )
