import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectFromModel
from sklearn.pipeline import Pipeline

class ModelFeatureSelector:

    def __init__(self, base_model, input_file_path, y_col, to_be_removed_cols):
        self.base_model = base_model
        self.df = pd.read_csv(input_file_path)
        self.X = self.df.drop(to_be_removed_cols, axis=1)
        self.y = self.df[y_col]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)


    def select_features(self):

        model = self.base_model

        # Verwendung von SelectFromModel, um Features basierend auf der Wichtigkeit auszuwählen
        selector = SelectFromModel(estimator=self.base_model)

        # Erstellen eines Pipelines mit LassoCV und SelectFromModel
        pipeline = Pipeline([
            ('feature_selection', selector),
            ('regression', model)
        ])

        # Anpassen des Modells
        pipeline.fit(self.X_train, self.y_train)

        # Nachdem das Modell angepasst wurde, können Sie die ausgewählten Features ermitteln
        selected_features = self.X_train.columns[pipeline.named_steps['feature_selection'].get_support()]

        not_selected_features = self.X_train.columns[~pipeline.named_steps['feature_selection'].get_support()]

        return not_selected_features



    def get_selected_features(self):
        pass

    def get_nonselected_features(self):
        pass