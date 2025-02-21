import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Tuple

@dataclass
class InferenceProcessor:
    model_choice: str

    def run(
        self,
        models: List[Any],
        data: pd.DataFrame,
    ) -> np.ndarray:
        prediction_function = self._get_model_prediction_function()
        predictions = np.mean(
            [prediction_function(model, data) for model in models], axis=0
        )

        return predictions

    def _get_model_prediction_function(self):
        if self.model_choice == "linear_regression":
            return lambda model, data: model.predict(data)
        elif self.model_choice == "gam":
            return lambda model, data: model.predict(data)
        elif self.model_choice == "gbm":
            return lambda model, data: model.predict(data)
        else:
            raise ValueError(f"Model choice {self.model_choice} not supported")
