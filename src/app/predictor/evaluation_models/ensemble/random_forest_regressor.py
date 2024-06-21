from sklearn.ensemble import RandomForestRegressor
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt

class RandomForestRegression():
        def __init__(self, filepath, n_estimators, random_state, max_depth, bootstrap, max_features):
            self.filepath = filepath
            self.model = RandomForestRegression(random_state=random_state, n_estimators=n_estimators, max_features=max_features, bootstrap=bootstrap, max_depth=max_depth)
            self.X_train, self.X_test, self.y_train, self.y_test, self.y_pred = [None] * 5

        def load_and_prepare_data(self):
            df = pd.read_csv(self.filepath)
            imputer = SimpleImputer(strategy='mean')
            df_imputed = pd.DataFrame(imputer.fit_transform(df.select_dtypes(include=['float64', 'int64'])))
            df_imputed.columns = df.select_dtypes(include=['float64', 'int64']).columns
            df_imputed['Timestamp'] = df['Timestamp']
            
            X = df_imputed[['cpu_pod_frontend', 'cpu_system', 'slower_memory', 'network_outgoing_system']]
            y = df_imputed['Durchschnittliche Antwortzeitintervalle']
            
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=42)


        def train_model(self):
            self.model.fit(self.X_train, self.y_train)
            self.y_pred = self.model.predict(self.X_test)


        def plot_results(self, hauptbereich_max=5000):
            plt.figure(figsize=(10, 6))
            im_hauptbereich = self.y_test <= hauptbereich_max
            plt.scatter(self.y_test[im_hauptbereich], self.y_pred[im_hauptbereich], alpha=0.5, label='Hauptbereich')
            ausreisser = self.y_test > hauptbereich_max
            plt.scatter(self.y_test[ausreisser], self.y_pred[ausreisser], color='red', alpha=0.5, label='Ausreißer')
            plt.ylim(0, hauptbereich_max)
            plt.xlim(0, hauptbereich_max)
            plt.title('Vorhersage vs. Tatsächliche Werte mit Fokus auf Hauptbereich')
            plt.xlabel('Tatsächliche Werte')
            plt.ylabel('Vorhersagewerte')
            plt.plot([0, hauptbereich_max], [0, hauptbereich_max], 'k--', lw=2)
            plt.legend()
            for i in self.y_test[ausreisser].index:
                plt.annotate(f'Ausreißer', (self.y_test[i], self.y_pred[i]), textcoords="offset points", xytext=(0,10), ha='center')
            plt.show()


        def evaluate(self):
             pass
