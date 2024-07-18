from abc import ABC, abstractmethod
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error
from sklearn.preprocessing import StandardScaler, MinMaxScaler

class Model(ABC):
    def __init__(self, filepath):
        self.filepath = filepath
        self.X_train, self.X_test, self.y_train, self.y_test, self.y_pred = [None] * 5
        self.model = self.create_model()

    @abstractmethod
    def create_model(self):
        pass


    def train_model(self):
        self.model.fit(self.X_train, self.y_train)
        self.y_pred = self.model.predict(self.X_test)


    
    def load_and_prepare_data(self, drop_features=True):
            df = pd.read_csv(self.filepath)
            imputer = SimpleImputer(strategy='mean')
            df_imputed = pd.DataFrame(imputer.fit_transform(df.select_dtypes(include=['float64', 'int64'])))
            df_imputed.columns = df.select_dtypes(include=['float64', 'int64']).columns
            df_imputed['Timestamp'] = df['Timestamp']

            if drop_features == True:
                features_to_be_dropped  = ['Durchschnittliche Antwortzeitintervalle', 'network_outgoing_pod-pod-1', 'network_outgoing_pod-pod-3', 'network_outgoing_pod-pod-7', 
                    'network_outgoing_pod-pod-8', 'network_outgoing_pod-pod-11','cpu_pod-pod-3', 'cpu_pod-pod-4',
                    'cpu_pod-pod-5', 'cpu_pod-pod-6', 'cpu_pod-pod-7', 'cpu_pod-pod-8', 'cpu_pod-pod-9', 'cpu_pod-pod-11',
                    'cpu_pod-pod-12', 'cpu_pod-pod-13', 'pod-restart-count-pod-1', 'pod-restart-count-pod-2',
                    'pod-restart-count-pod-3', 'pod-restart-count-pod-4', 'pod-restart-count-pod-5', 'pod-restart-count-pod-6', 'pod-restart-count-pod-7',
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
            elif drop_features == "remove_pod_metrics": 
                features_to_be_dropped  = ['Durchschnittliche Antwortzeitintervalle', 'Timestamp',
                    'network_outgoing_pod-pod-1', 'network_outgoing_pod-pod-2', 'network_outgoing_pod-pod-3',
                    'network_outgoing_pod-pod-4', 'network_outgoing_pod-pod-5', 'network_outgoing_pod-pod-6', 
                    'network_outgoing_pod-pod-7', 'network_outgoing_pod-pod-8', 'network_outgoing_pod-pod-9', 
                    'network_outgoing_pod-pod-10', 'network_outgoing_pod-pod-11', 'cpu_pod-pod-3', 'cpu_pod-pod-4', 
                    'cpu_pod-pod-1', 'cpu_pod-pod-2', 'cpu_pod-pod-5', 'cpu_pod-pod-6', 'cpu_pod-pod-7', 'cpu_pod-pod-8', 'cpu_pod-pod-9', 'cpu_pod-pod-10',
                    'cpu_pod-pod-11', 'cpu_pod-pod-12', 'cpu_pod-pod-13', 'pod-restart-count-pod-1', 'pod-restart-count-pod-2',
                    'pod-restart-count-pod-3', 'pod-restart-count-pod-4',  'pod-restart-count-pod-5', 'pod-restart-count-pod-6', 
                    'pod-restart-count-pod-7', 'cpu_pod-pod-1_rolling_avg_5', 'cpu_pod-pod-11_rolling_avg_5', 
                    'cpu_pod-pod-8_rolling_avg_5', 'network_outgoing_system',
                    'pod-restart-count-pod-8', 'pod-restart-count-pod-9', 'pod-restart-count-pod-10',
                    'pod-restart-count-pod-11', 'pod-restart-count-pod-12', 'pod-restart-count-pod-14',
                    'pod-restart-count-pod-13', 'pod-restart-count-pod-15',
                    'pod-restart-count-pod-16', 'pod-restart-count-pod-17',
                    'pod-restart-count-pod-18', 'pod-restart-count-pod-19', 'pod-restart-count-pod-25',
                    'pod-restart-count-pod-20', 'pod-restart-count-pod-18', 'pod-restart-count-pod-19', 
                    'pod-restart-count-pod-20', 'pod-restart-count-pod-24',
                    'pod-restart-count-pod-21', 'pod-restart-count-pod-22', 'pod-restart-count-pod-23','network_outgoing_pod-pod-14', 'cpu_pod-pod-14',
                    'network_outgoing_pod-pod-12', 'network_outgoing_pod-pod-13', 'network_outgoing_pod-pod-14',
                    'network_outgoing_pod-pod-15', 'network_outgoing_pod-pod-16',
                    'network_outgoing_pod-pod-17', 'network_outgoing_pod-pod-18',
                    'network_outgoing_pod-pod-19', 'network_outgoing_pod-pod-20',
                    'network_outgoing_pod-pod-21', 'network_outgoing_pod-pod-22',
                    'network_outgoing_pod-pod-23', 'network_outgoing_pod-pod-24',
                    'network_outgoing_pod-pod-25', 'cpu_pod-pod-15', 'cpu_pod-pod-16',
                    'cpu_pod-pod-17', 'cpu_pod-pod-18', 'cpu_pod-pod-19', 'cpu_pod-pod-20',
                    'cpu_pod-pod-21', 'cpu_pod-pod-22', 'cpu_pod-pod-23', 'cpu_pod-pod-24',
                    'cpu_pod-pod-25']
                X = df_imputed.drop(features_to_be_dropped, axis=1)

            else:
                features_to_be_dropped  = ['Durchschnittliche Antwortzeitintervalle', 'Timestamp',
                    'network_outgoing_pod-pod-1', 'network_outgoing_pod-pod-2', 'network_outgoing_pod-pod-3',
                    'network_outgoing_pod-pod-4', 'network_outgoing_pod-pod-5', 'network_outgoing_pod-pod-6', 'network_outgoing_pod-pod-7', 
                    'network_outgoing_pod-pod-8', 'network_outgoing_pod-pod-11', 'cpu_pod-pod-3', 'cpu_pod-pod-4',
                    'cpu_pod-pod-5', 'cpu_pod-pod-6', 'cpu_pod-pod-7', 'cpu_pod-pod-8', 'cpu_pod-pod-9', 'cpu_pod-pod-10', 'cpu_pod-pod-11',
                    'cpu_pod-pod-12', 'cpu_pod-pod-13', 'pod-restart-count-pod-1', 'pod-restart-count-pod-2',
                    'pod-restart-count-pod-3', 'pod-restart-count-pod-4',  'pod-restart-count-pod-5', 'pod-restart-count-pod-6',  'pod-restart-count-pod-7',
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
                    'cpu_pod-pod-25', ]
                X = df_imputed.drop(features_to_be_dropped, axis=1)
            
            print(X)
            y = df_imputed['Durchschnittliche Antwortzeitintervalle']
            
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.3, random_state=42)


    def prepare_for_svr(self):
        df = pd.read_csv(self.filepath)
        imputer = SimpleImputer(strategy='mean')
        df_imputed = pd.DataFrame(imputer.fit_transform(df.select_dtypes(include=['float64', 'int64'])))
        df_imputed.columns = df.select_dtypes(include=['float64', 'int64']).columns
        df_imputed['Timestamp'] = df['Timestamp']

        X = df_imputed[['cpu_system', 'memory', 'network_outgoing_system' ,'cpu_pod-pod-7', 'cpu_pod-pod-8', 'i_o_read', 'i_o_write']]

        scaler = StandardScaler()
        X = scaler.fit_transform(X)
            
        y = df_imputed['Durchschnittliche Antwortzeitintervalle']

        y = y.values.reshape(-1, 1)

        # Für Standardisierung

        y_scaled = scaler.fit_transform(y)
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y_scaled, test_size=0.3, random_state=42)


    def prepare_without_pod_level_metrics(self):
        df = pd.read_csv(self.filepath)
        imputer = SimpleImputer(strategy='mean')
        df_imputed = pd.DataFrame(imputer.fit_transform(df.select_dtypes(include=['float64', 'int64'])))
        df_imputed.columns = df.select_dtypes(include=['float64', 'int64']).columns
        df_imputed['Timestamp'] = df['Timestamp']

        X = df_imputed[['cpu_system', 'memory', 'network_outgoing_system' ,'i_o_read', 'i_o_write']]
            
        y = df_imputed['Durchschnittliche Antwortzeitintervalle']

        y = y.values.reshape(-1, 1)

        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.3, random_state=42)



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

    def evaluate_svr_model(self):
        mse = mean_squared_error(self.y_test, self.y_pred)
        print("MSE:", mse)

        # Berechnung des RMSE (Root Mean Squared Error)
        rmse = np.sqrt(mse)
        print("RMSE:", rmse)

        # Berechnung des MAE (Mean Absolute Error)
        mae = mean_absolute_error(self.y_test, self.y_pred)
        print("MAE:", mae)

        r2 = r2_score(self.y_test, self.y_pred)
        print("R²-Score:", r2)

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
         

    def plot_results(self, hauptbereich_max=5000):
        plt.figure(figsize=(10, 6))
        im_hauptbereich = self.y_test <= hauptbereich_max
        plt.scatter(self.y_test[im_hauptbereich], self.y_pred[im_hauptbereich], alpha=0.5, label='Hauptbereich')
        ausreisser = self.y_test > hauptbereich_max
        plt.scatter(self.y_test[ausreisser], self.y_pred[ausreisser], color='red', alpha=0.5, label='Ausreißer')
        plt.ylim(0, hauptbereich_max)
        plt.xlim(0, hauptbereich_max)
        plt.title('Predicted  vs. actual values with focus on main main area')
        plt.xlabel('Actual values')
        plt.ylabel('Predicted values')
        plt.plot([0, hauptbereich_max], [0, hauptbereich_max], 'k--', lw=2)
        plt.legend()
        for i in self.y_test[ausreisser].index:
            plt.annotate(f'Ausreißer', (self.y_test[i], self.y_pred[i]), textcoords="offset points", xytext=(0,10), ha='center')
        plt.show()

    def evaluate(self):
            pass