from models.base_model import BaseModel
from statsmodels.tsa.arima.model import ARIMA
import numpy as np
import itertools
import warnings

class ARIMA_Model(BaseModel):
    def __init__(self, p=None, d=None, q=None):
        self.p, self.d, self.q = p, d, q
        self.model_fit = None
        self.best_order = None
        self.original_data = None

    def fit(self, data):
        self.original_data = list(data)
        if self.p is not None:
            self.model_fit = ARIMA(self.original_data, order=(self.p, self.d, self.q)).fit()
            self.best_order = (self.p, self.d, self.q)
        else:
            self.best_order, self.model_fit = self._auto_tune()
            self.p, self.d, self.q = self.best_order


    def predict(self, horizon):
        forecast = self.model_fit.get_forecast(steps=horizon)
        return forecast.predicted_mean, np.sqrt(forecast.var_pred_mean)

    def append(self, new_data):
        if isinstance(new_data, (list, np.ndarray)):
            for val in new_data:
                self.model_fit = self.model_fit.append([val])
                self.original_data.append(val)
        elif isinstance(new_data, (int, float)):
            self.model_fit = self.model_fit.append([new_data])
            self.original_data.append(new_data)

    def reset(self):
        if self.original_data and self.best_order:
            self.model_fit = ARIMA(self.original_data, order=self.best_order).fit()

    def _auto_tune(self):
        p_range, d_range, q_range = range(0, 4), range(0, 2), range(0, 4)
        best_aic = float('inf')
        best_order = None
        best_model = None
        warnings.filterwarnings("ignore")
        for order in itertools.product(p_range, d_range, q_range):
            try:
                model = ARIMA(self.original_data, order=order)
                fit = model.fit()
                if fit.aic < best_aic:
                    penalty = 0 if (order[0] > 0 or order[2] > 1) else 100
                    if fit.aic + penalty < best_aic:
                        best_aic = fit.aic + penalty
                        best_order, best_model = order, fit
            except Exception as e: 
                print(f"Error fitting ARIMA model with order {order}: {e}")
        warnings.resetwarnings()
        return best_order, best_model
    
    def get_params(self):
        return {'p': self.p, 'd': self.d, 'q': self.q}
    
    def get_num_params(self) -> int:
        """
        Returns the number of parameters in the fitted ARIMA model.
        If the model is not fitted, returns 0.
        """
        return len(self.model_fit.params) if self.model_fit else 0