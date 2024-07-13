from src.app.predictor.contracts.model import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from scikeras.wrappers import KerasClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
import pandas as pd
import numpy as np
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error

class SequentialNetwork():
    def __init__(self, filepath, loss_function, epochs, validation_split, learning_rate):
        self.loss_function = loss_function
        self.epochs = epochs
        self.validation_split = validation_split
        self.learning_rate = learning_rate
        self.filepath = filepath
        self.X_train, self.X_test, self.y_train, self.y_test, self.y_pred = [None] * 5

    def create_model(self):
        return Sequential([
            Dense(64, activation='relu', input_shape=(self.X_train.shape[1],)),  
            Dense(64, activation='relu'), 
            Dropout(0.2),
            Dense(1)
        ])
    
    def compile_model(self):
        model = self.create_model()
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(self.X_train)
        X_test_scaled = scaler.transform(self.X_test)
        model.compile(optimizer=Adam(learning_rate=self.learning_rate), loss='mean_squared_error')
        model.fit(X_train_scaled, self.y_train, validation_split=self.validation_split, epochs=self.epochs)
        self.y_pred = model.predict(X_test_scaled)

    def load_and_prepare_data(self, drop_features=True):
        df = pd.read_csv(self.filepath)
        imputer = SimpleImputer(strategy='mean')
        df_imputed = pd.DataFrame(imputer.fit_transform(df.select_dtypes(include=['float64', 'int64'])))
        df_imputed.columns = df.select_dtypes(include=['float64', 'int64']).columns
        df_imputed['Timestamp'] = df['Timestamp']

        if drop_features == True:
            features_to_be_dropped  = ['Requests je Sekunde','Durchschnittliche Antwortzeitintervalle','network_outgoing_pod-pod-1', 'network_outgoing_pod-pod-3',
                'network_outgoing_pod-pod-4', 'network_outgoing_pod-pod-5', 'network_outgoing_pod-pod-7', 'network_outgoing_pod-pod-8',
                'network_outgoing_pod-pod-9', 'network_outgoing_pod-pod-10',
                'network_outgoing_pod-pod-11', 
                'cpu_pod-pod-1', 'cpu_pod-pod-2', 'cpu_pod-pod-3', 'cpu_pod-pod-4',
                'cpu_pod-pod-5', 'cpu_pod-pod-6', 'cpu_pod-pod-7', 'cpu_pod-pod-8', 'cpu_pod-pod-9', 'cpu_pod-pod-11',
                'cpu_pod-pod-12', 'cpu_pod-pod-13', 'pod-restart-count-pod-1', 'pod-restart-count-pod-2',
                'pod-restart-count-pod-3', 'pod-restart-count-pod-4',
                'pod-restart-count-pod-5', 'pod-restart-count-pod-6',
                'pod-restart-count-pod-8', 'pod-restart-count-pod-10',
                'pod-restart-count-pod-11', 'pod-restart-count-pod-12',
                'pod-restart-count-pod-13', 'network_outgoing_pod-pod-14', 'cpu_pod-pod-14',
                'network_outgoing_pod-pod-15', 'network_outgoing_pod-pod-16',
                'network_outgoing_pod-pod-17', 'network_outgoing_pod-pod-18',
                'network_outgoing_pod-pod-19', 'network_outgoing_pod-pod-20',
                'network_outgoing_pod-pod-21', 'network_outgoing_pod-pod-22',
                'network_outgoing_pod-pod-23', 'network_outgoing_pod-pod-24',
                'network_outgoing_pod-pod-25', 'cpu_pod-pod-15', 'cpu_pod-pod-16',
                'cpu_pod-pod-17', 'cpu_pod-pod-18', 'cpu_pod-pod-19', 'cpu_pod-pod-20',
                'cpu_pod-pod-21', 'cpu_pod-pod-22', 'cpu_pod-pod-23', 'cpu_pod-pod-24',
                'cpu_pod-pod-25', 'pod-restart-count-pod-14',
                'pod-restart-count-pod-15', 'pod-restart-count-pod-16',
                'pod-restart-count-pod-17', 'pod-restart-count-pod-18',
                'pod-restart-count-pod-19', 'pod-restart-count-pod-20',
                'pod-restart-count-pod-21', 'pod-restart-count-pod-22',
                'pod-restart-count-pod-23', 'pod-restart-count-pod-24',
                'pod-restart-count-pod-25']
            X = df_imputed.drop(features_to_be_dropped, axis=1)
        else: 
            features_to_be_dropped  = ['Durchschnittliche Antwortzeitintervalle',
                'network_outgoing_pod-pod-1', 'network_outgoing_pod-pod-2', 'network_outgoing_pod-pod-3',
                'network_outgoing_pod-pod-4', 'network_outgoing_pod-pod-5', 'network_outgoing_pod-pod-6', 'network_outgoing_pod-pod-7', 
                'network_outgoing_pod-pod-8', 'network_outgoing_pod-pod-9', 'network_outgoing_pod-pod-10',
                'network_outgoing_pod-pod-11','cpu_pod-pod-1','cpu_pod-pod-2', 'cpu_pod-pod-3', 'cpu_pod-pod-4',
                'cpu_pod-pod-5', 'cpu_pod-pod-6', 'cpu_pod-pod-7', 'cpu_pod-pod-8', 'cpu_pod-pod-9', 'cpu_pod-pod-10', 'cpu_pod-pod-11',
                'cpu_pod-pod-12', 'cpu_pod-pod-13', 'pod-restart-count-pod-1', 'pod-restart-count-pod-2',
                'pod-restart-count-pod-3', 'pod-restart-count-pod-4',
                'pod-restart-count-pod-5', 'pod-restart-count-pod-6',  'pod-restart-count-pod-7',
                'pod-restart-count-pod-8', 'pod-restart-count-pod-9', 'pod-restart-count-pod-10',
                'pod-restart-count-pod-11', 'pod-restart-count-pod-12',
                'pod-restart-count-pod-13', 'network_outgoing_pod-pod-14', 'cpu_pod-pod-14',
                'network_outgoing_pod-pod-12', 'network_outgoing_pod-pod-13', 'network_outgoing_pod-pod-14',
                'network_outgoing_pod-pod-15', 'network_outgoing_pod-pod-16',
                'network_outgoing_pod-pod-17', 'network_outgoing_pod-pod-18',
                'network_outgoing_pod-pod-19', 'network_outgoing_pod-pod-20',
                'network_outgoing_pod-pod-21', 'network_outgoing_pod-pod-22',
                'network_outgoing_pod-pod-23', 'network_outgoing_pod-pod-24',
                'network_outgoing_pod-pod-25', 'cpu_pod-pod-15', 'cpu_pod-pod-16',
                'cpu_pod-pod-17', 'cpu_pod-pod-18', 'cpu_pod-pod-19', 'cpu_pod-pod-20',
                'cpu_pod-pod-21', 'cpu_pod-pod-22', 'cpu_pod-pod-23', 'cpu_pod-pod-24',
                'cpu_pod-pod-25', 'pod-restart-count-pod-14',
                'pod-restart-count-pod-15', 'pod-restart-count-pod-16',
                'pod-restart-count-pod-17', 'pod-restart-count-pod-18',
                'pod-restart-count-pod-19', 'pod-restart-count-pod-20',
                'pod-restart-count-pod-21', 'pod-restart-count-pod-22',
                'pod-restart-count-pod-23', 'pod-restart-count-pod-24',
                'pod-restart-count-pod-25']
            X = df_imputed.drop(features_to_be_dropped, axis=1)
        
        
        y = df_imputed['Durchschnittliche Antwortzeitintervalle']
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    def custom_mape(self, y_true, y_pred):
        # Überprüfen, ob y_true und y_pred Pandas Series sind, und in numpy Arrays umwandeln
        if isinstance(y_true, pd.Series):
            y_true = y_true.to_numpy()
        if isinstance(y_pred, pd.Series):
            y_pred = y_pred.to_numpy()
        
        # Filtern der Paare, bei denen y_true nicht 0 ist
        non_zero_indices = np.where(y_true != 0)
        
        # Anwenden des Filters auf die Arrays
        y_true_filtered = y_true[non_zero_indices]
        y_pred_filtered = y_pred[non_zero_indices]
        
        # Berechnung des MAPE nur für Nicht-Null Werte
        mape = np.mean(np.abs((y_true_filtered - y_pred_filtered) / y_true_filtered))
        return mape

    def evaluate_model(self):
        mse = mean_squared_error(self.y_test, self.y_pred)
        print("MSE:", mse)

        # Berechnung des RMSE (Root Mean Squared Error)
        rmse = np.sqrt(mse)
        print("RMSE:", rmse)

        # Berechnung des MAE (Mean Absolute Error)
        mae = mean_absolute_error(self.y_test, self.y_pred)
        print("MAE:", mae)

        # Berechnung des MAPE (Mean Absolute Percentage Error)
        # Achtung: MAPE erwartet, dass keine echten Werte von 0 enthalten sind, da dies zu einer Division durch Null führen würde.
        mape = self.custom_mape(self.y_test, self.y_pred)
        print(mape)

        r2 = r2_score(self.y_test, self.y_pred)
        print("R²-Score:", r2)
        

    def build_model(self, learning_rate=0.01, dropout_rate=0.2, n_neurons=64):
        model = Sequential([
            Dense(n_neurons, activation='relu', input_shape=(self.X_train.shape[1],)),
            Dense(n_neurons, activation='relu'),
            Dropout(dropout_rate),
            Dense(1)
        ])
        model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mean_squared_error')
        return model
    

    def model_wrapper(self, learning_rate, dropout_rate, n_neurons):
        def create_model():
            model = Sequential([
                Dense(n_neurons, activation='relu', input_shape=(self.X_train.shape[1],)),
                Dense(n_neurons, activation='relu'),
                Dropout(dropout_rate),
                Dense(1)
            ])
            model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mean_squared_error')
            return model
        return create_model

    def search_for_best_model(self):
        # Wrapper-Funktion als build_fn verwenden
        model = KerasClassifier(build_fn=lambda: self.model_wrapper(learning_rate=0.01, dropout_rate=0.2, n_neurons=64), epochs=100, batch_size=10, verbose=0)
        param_dist = {
            'learning_rate': [0.01, 0.001, 0.0001],
            'dropout_rate': [0.2, 0.3, 0.4],
            'n_neurons': [64, 128, 256]
        }
        random_search = RandomizedSearchCV(estimator=model, param_distributions=param_dist, n_iter=10, cv=3)
        random_search_result = random_search.fit(self.X_train, self.y_train)

        print("Best: %f using %s" % (random_search_result.best_score_, random_search_result.best_params_))
    
    
    