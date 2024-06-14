from sklearn.linear_model import Lasso  # Beachte, dass in neueren Versionen der Name geändert sein könnte
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.decomposition import PCA

class FeatureSelector:
    def __init__(self, path_to_prepared_and_merged_csv):
        self.dataset = pd.read_csv(path_to_prepared_and_merged_csv)
        # Optional: Entferne Spalten, die nicht für die Feature Selection verwendet werden sollen
        # z.B. Timestamp oder nicht-numerische Spalten
        self.dataset.drop(['Timestamp'], axis=1, inplace=True)

    def select_features(self):
        # Trenne Features und Zielvariable, falls nötig
        # Angenommen, 'Durchschnittliche Antwortzeitintervalle' ist die Zielvariable
        X = self.dataset.drop('Durchschnittliche Antwortzeitintervalle', axis=1)
        y = self.dataset['Durchschnittliche Antwortzeitintervalle']

        # Skaliere die Features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Initialisiere und trainiere den Randomized Lasso
        # Beachte: Hier als Platzhalter Lasso verwendet. Ersetze dies durch RandomizedLasso oder die entsprechende Klasse
        model = Lasso()
        model.fit(X_scaled, y)
        
        # Identifiziere und drucke die ausgewählten Features
        selected_features = X.columns[(model.coef_ != 0)]
        print("Ausgewählte Features:", selected_features)

    def apply_pca(self, n_components=None, explained_variance_threshold=None):
        # Trenne Features und Zielvariable, falls nötig
        X = self.dataset.drop('Durchschnittliche Antwortzeitintervalle', axis=1)
        y = self.dataset['Durchschnittliche Antwortzeitintervalle']
        
        # Skaliere die Features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Initialisiere PCA
        if n_components is not None:
            pca = PCA(n_components=n_components)
        elif explained_variance_threshold is not None:
            pca = PCA(n_components=0.95)  # Standardwert; wird später angepasst
        else:
            raise ValueError("Entweder n_components oder explained_variance_threshold muss angegeben werden")

        # Führe PCA aus
        X_pca = pca.fit_transform(X_scaled)

        # Wenn ein Schwellenwert für die erklärte Varianz angegeben wurde, passe n_components entsprechend an
        if explained_variance_threshold is not None:
            cumulative_variance = pca.explained_variance_ratio_.cumsum()
            n_components = (cumulative_variance < explained_variance_threshold).sum() + 1
            pca = PCA(n_components=n_components)
            X_pca = pca.fit_transform(X_scaled)
        
        print(f"Anzahl der ausgewählten Hauptkomponenten: {n_components}")
        return X_pca, y


