from sklearn.ensemble import GradientBoostingRegressor
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
from src.app.predictor.contracts.model import Model

class GradientBoostingRegression(Model):
    def __init__(self, filepath, n_estimators, random_state, max_depth, max_features):
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.max_depth = max_depth
        self.max_features = max_features
        super().__init__(filepath)

    def create_model(self):
        return GradientBoostingRegressor(random_state=self.random_state, n_estimators=self.n_estimators, max_features=self.max_features, max_depth=self.max_depth)


class RandomForestRegressor(Model):
    def __init__(self, filepath, n_estimators, random_state, max_depth, max_features):
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.max_depth = max_depth
        self.max_features = max_features
        super().__init__(filepath)

    def create_model(self):
        return RandomForestRegressor(random_state=self.random_state, n_estimators=self.n_estimators, max_features=self.max_features, max_depth=self.max_depth)


        


    
