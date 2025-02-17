import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
from dlm_model import DLM


class HierarchicalTimeSeries:

    def __init__(self, data: Dict[str, pd.Series], hierarchical_structure: Dict[str, List[str]], **kwargs):
        """
        Initialize the hierarchical time series model
        
        Args:
        data: A dictionary of pandas series with the keys being the model / series id
        hierarchical_structure: A dictionary with the hierarchical structure of the time series. Includes modelsÑ_ids
        kwargs: A dictionary with the model params for each model
            model_type: The type of model to use
            params: The parameters for the model in a tuple"""


        self.data = data
        self.hierarchical_structure = hierarchical_structure    
        self.models = {}    # We identify all time series with an id
        self.forecasts: Dict[str, List[int]] = {}   # Makes each of the predictions for the models for 1t step
        self.model_types = {}   # The model types for each of the models

        for model_id, series in data.items():
            model_type = kwargs.get("model_type")[model_id]
            model_params = kwargs.get("model_params")[model_id]
            self.model_types[model_id] = model_type
            self.models[model_id] = DLM(**model_params)
        

    def predict(self, model_id, series, model_type, **model_params):
        """
        This function does 1 step t for the model
        """
        # Add here the different model types
        model_type = self.model_types[model]
        model = self.models[model_id]
        y = self.data[series]

        if model_type == "DLM":
            pred_mean, pred_cov = model.predict()
            self.forecasts[model_id] = pred_mean[0, 0]
            model.update(y)

            return pred_mean[0, 0], pred_cov[0, 0]

        else:
            raise ValueError("Model type not supported")
        
    
    def predict_models(self,**model_params):
        """
        Fit all models to the data
        """
        # We make recursive calls to fit the models
        for model_id in self.models.keys():
            self.predict(model_id, **model_params)

    def reconcile_forecasts(self, method: str):
        """
        Reconcile forecasts
        """

        if method == "top_down":
            total_forecast = self.forecasts.get("total")

            if total_forecast is None:
                total_forecast = sum(self.forecasts.values())
                self.forecasts["total"] = total_forecast

            for node in self.hierarchical_structure["hierarchy"]:
                if node in self.forecasts:
                    proportion = self.forecasts[node] / total_forecast
                    self.forecasts[node] = proportion * total_forecast
                
        elif method == "bottom_up":
            # We sum the forecasts of the bottom levels
            total_forecast = 0
            
            # Recursively sum the forecasts of the bottom levels

            def sum_bottom_forecasts(node):
                if node in self.hierarchical_structure["bottom"]:
                    return self.forecasts[node]
                else:
                    return sum(sum_bottom_forecasts(child) for child in self.hierarchical_structure[node])
            
            total_forecast = sum_bottom_forecasts("total")

            for node in self.hierarchical_structure["total"]:
                if node in self.forecasts:
                    total_forecast += self.forecasts[node]

        elif method == "middle_out":
            # We sum the forecasts of the middle levels
            total_forecast = 0

            # Recursively sum the forecasts of the middle levels

            def sum_middle_forecasts(node):
                if node in self.hierarchical_structure["middle"]:
                    return self.forecasts[node]
                else:
                    return sum(sum_middle_forecasts(child) for child in self.hierarchical_structure[node])
            
            total_forecast = sum_middle_forecasts("total")

            for node in self.hierarchical_structure["total"]:
                if node in self.forecasts:
                    total_forecast += self.forecasts[node]
