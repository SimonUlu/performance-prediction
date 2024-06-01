import pandas as pd
from sklearn.model_selection import train_test_split

class Preprocessor:
    def __init__(self, filepath, y_label, drop_columns=None, test_size=0.2, random_state=42):
        self.filepath = filepath
        self.y_label = y_label
        self.drop_columns = drop_columns if drop_columns is not None else []
        self.test_size = test_size
        self.random_state = random_state
        self.X_train = pd.DataFrame()
        self.X_test = pd.DataFrame()
        self.y_train = pd.Series() 
        self.y_test = pd.Series()  


    def preprocess(self):
        try:
            df = pd.read_csv(self.filepath)
        except Exception as e:
            raise FileNotFoundError(f"Die Datei unter {self.filepath} konnte nicht gefunden oder gelesen werden: {e}")

        if not set(self.drop_columns).issubset(df.columns):
            raise ValueError("Ein oder mehrere Spalten zum Entfernen existieren nicht im DataFrame.")

        if self.y_label not in df.columns:
            raise ValueError(f"Die Zielvariable '{self.y_label}' existiert nicht im DataFrame.")

        df = df.drop(self.drop_columns, axis=1)

        X = df.drop(self.y_label, axis=1)
        y = df[self.y_label]

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state)
        

    



