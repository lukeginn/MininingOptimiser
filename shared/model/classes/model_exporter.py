import logging as logger
import joblib
from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Tuple

@dataclass
class ModelExporter:

    def save(self, models: List[Any], path: str, path_suffix: str) -> None:
        for i, model in enumerate(models):
            full_file_path = f"{path}\\model_{path_suffix}_{i}.pkl"
            joblib.dump(model, full_file_path)
            logger.info(f"Model saved to {full_file_path}")

    def load(self, file_paths: List[str]) -> List[Any]:
        models = [joblib.load(file_path) for file_path in file_paths]
        logger.info(f"Models loaded from {file_paths}")
        return models
