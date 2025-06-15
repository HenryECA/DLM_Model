from models.base_model import BaseModel

class SMA(BaseModel):
    def __init__(self, window_size):
        self.window_size = window_size

    def fit(self, train_seq, *args, **kwargs):
        pass

    def predict(self, horizon, x):
        preds = []
        for _ in range(horizon):
            sma = sum(x) / len(x)
            preds.append(sma)
            x[1:].append(sma)
        return preds
    
    def get_num_params(self):
        return 1
    
    def get_params(self):
        return {'window_size': self.window_size}
