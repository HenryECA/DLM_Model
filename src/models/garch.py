from arch import arch_model
import numpy as np
import itertools
import warnings
from models.base_model import BaseModel

class GARCH_Model(BaseModel):
    def __init__(self, p=None, q=None):
        self.p, self.q = p, q
        self.model_fit = None
        self.best_order = None
        self.original_data = None

    def fit(self, data):
        self.original_data = list(data)
        if self.p is not None:
            # Fit with specified order
            self.model_fit = arch_model(
                self.original_data,
                vol='GARCH', p=self.p, q=self.q
            ).fit(disp='off')
            self.best_order = (self.p, self.q)
        else:
            # Auto-select best order
            self.best_order, self.model_fit = self._auto_tune()

    def predict(self, horizon):
        # Generate h-step ahead forecasts
        forecast = self.model_fit.forecast(horizon=horizon)
        # Mean forecast for the last available period
        mean_forecast = forecast.mean.values[-1]
        # Variance forecast for the last available period
        var_forecast = forecast.variance.values[-1]

        return mean_forecast, var_forecast

    def append(self, new_data):
        # Append new observations and re-fit
        if isinstance(new_data, (list, np.ndarray)):
            for val in new_data:
                self.original_data.append(val)
                self.model_fit = arch_model(
                    self.original_data,
                    vol='GARCH', p=self.best_order[0], q=self.best_order[1]
                ).fit(disp='off')
        elif isinstance(new_data, (int, float)):
            self.original_data.append(new_data)
            self.model_fit = arch_model(
                self.original_data,
                vol='GARCH', p=self.best_order[0], q=self.best_order[1]
            ).fit(disp='off')

    def reset(self):
        # Re-fit model using original data and best order
        if self.original_data and self.best_order:
            self.model_fit = arch_model(
                self.original_data,
                vol='GARCH', p=self.best_order[0], q=self.best_order[1]
            ).fit(disp='off')

    def _auto_tune(self):
        # Grid search over p and q parameters
        p_range, q_range, o_range = range(1, 4), range(1, 4), range(0, 4)
        best_aic = np.inf
        best_order = None
        best_model = None
        warnings.filterwarnings('ignore')
        # Generate all combinations of p, q and o
        for order in itertools.product(p_range, q_range):
            try:
                model = arch_model(
                    self.original_data,
                    vol='GARCH', p=order[0], q=order[1]
                )
                fit = model.fit(disp='off')
                if fit.aic < best_aic:
                    best_aic = fit.aic
                    best_order = order
                    best_model = fit
            except Exception as e:
                print(f"Error fitting GARCH model with order {order}: {e}")
        warnings.resetwarnings()
        return best_order, best_model
    
    def get_params(self):
        return {'p': self.p, 'q': self.q}
    
    def get_num_params(self) -> int:
        """
        Returns the number of parameters in the GARCH model.
        The number of parameters is p + q + 1 (for the constant term).
        """
        return self.p + self.q + 1 if self.p is not None and self.q is not None else 0

