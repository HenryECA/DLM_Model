from abc import ABC, abstractmethod

class BaseModel(ABC):
    @abstractmethod
    def fit(self, train_data): pass

    @abstractmethod
    def predict(self, horizon): pass

    def append(self, new_data): pass

    def reset(self): pass