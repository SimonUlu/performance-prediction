from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
from src.app.predictor.contracts.model import Model
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR
import xgboost as xgb

class GradientBoostingRegression(Model):
    def __init__(self, filepath, n_estimators, random_state, max_depth, max_features):
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.max_depth = max_depth
        self.max_features = max_features
        super().__init__(filepath)

    def create_model(self):
        return GradientBoostingRegressor(random_state=self.random_state, n_estimators=self.n_estimators, max_features=self.max_features, max_depth=self.max_depth)


class RandomForestRegression(Model):
    def __init__(self, filepath, n_estimators, random_state, max_depth, max_features, param_grid=None):
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.max_depth = max_depth
        self.max_features = max_features
        # Standard-Parameterraster, wenn keines angegeben wird
        if param_grid is None:
            self.param_grid = {
                'n_estimators': [100, 200, 500],
                'max_depth': [10, 20, 30, None],
                'bootstrap': [True, False],
                'max_features': ['auto', 0.5, 0.75, 1.0]
            }
        else:
            self.param_grid = param_grid
        super().__init__(filepath)

    def find_best_model(self, X_train, y_train):
        model = RandomForestRegressor(random_state=42)
        
        grid_search = GridSearchCV(estimator=model, param_grid=self.param_grid, cv = 3, n_jobs=-1, verbose=2)

        grid_search.fit(X_train, y_train)

        print(grid_search.best_params_)

        return grid_search.best_params_

    def create_model(self):
        return RandomForestRegressor(random_state=self.random_state, n_estimators=self.n_estimators, max_features=self.max_features, max_depth=self.max_depth)

class SVRRegressor(Model):

    def __init__(self, filepath):
        super().__init__(filepath)

    def create_model(self):
        return SVR()
    
class XGBoostRegressor(Model):
    def __init__(self, filepath, colsample_bytree, learning_rate, max_depth, alpha, n_estimators):
        self.colsample_bytree = colsample_bytree
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.alpha = alpha
        self.n_estimators = n_estimators
        super().__init__(filepath)

    def create_model(self):
        return xgb.XGBRegressor(objective='reg:squarederror', 
                                colsample_bytree = 0.3, 
                                learning_rate = self.learning_rate, 
                                max_depth = self.max_depth, 
                                alpha = self.alpha, 
                                n_estimators = self.n_estimators)



      


    
