from src.app.predictor.contracts.model import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input, Dropout
from tensorflow.keras.optimizers import Adam

class SeuqentialNetwork(Model):
    def __init__(self, filepath, loss_function, shape):
        self.loss_function = loss_function
        self.shape = shape
        super().__init__(filepath)

    def create_model(self):
        return Sequential([
            Dense(64, activation='relu', input_shape=(self.shape)),
            Dense(64, activation='relu'),
            Dropout(0.2),
            Dense(32, activation='relu'),
            Dropout(0.2),
            Dense(1)
        ])
