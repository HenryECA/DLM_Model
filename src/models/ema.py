from models.base_model import BaseModel

class EMA(BaseModel):
    def __init__(self, alpha):
        self.alpha = alpha
        self.train_data = None
        self.last_ema = None

    def fit(self, train_seq, *args, **kwargs):
        # Flatten sequences and calculate the last EMA from training data
        values = [label[0] for _, label in train_seq]
        ema = values[0]
        for val in values[1:]:
            ema = self.alpha * val + (1 - self.alpha) * ema
        self.last_ema = ema

    def predict(self, horizon, last_sequence):
        preds = []
        ema = self.last_ema
        for _ in range(horizon):
            ema = self.alpha * ema + (1 - self.alpha) * ema  # EMA predicts using itself
            preds.append(ema)
        
        # Update the last EMA for the next prediction using the last sequence
        self.last_ema = self.alpha * last_sequence[-1] + (1 - self.alpha) * self.last_ema
        return preds

    def get_num_params(self):
        return 1
    
    def get_params(self):
        return {'alpha': self.alpha}