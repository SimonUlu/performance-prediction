from sklearn.feature_selection import RFECV
import pandas as pd
from sklearn.model_selection import train_test_split

class RecursiveFeatureSelector:

    def __init__(self, base_model, input_file_path, y_col, to_be_removed_cols):
        self.base_model = base_model
        self.df = pd.read_csv(input_file_path)
        self.X = self.df.drop(to_be_removed_cols, axis=1)
        self.y = self.df[y_col]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)


    def select_features(self, step, cross_validation_folds):
        estimator = self.base_model

        # Feature Selection
        selector = RFECV(estimator, step=step, cv=cross_validation_folds)
        selector = selector.fit(self.X_train, self.y_train)

        print("Optimale Anzahl von Features : %d" % selector.n_features_)

        selected_features = self.X_train.columns[selector.support_]

        not_selected_features = self.X_train.columns[~selector.support_]

        return not_selected_features
    
    

    def get_most_important_features_ordered(self, number):
        model = self.base_model

        model.fit(self.X_train, self.y_train)

        # Holen der Feature Importances
        importances = model.feature_importances_

        # Konvertieren der Importances in einen DataFrame für eine bessere Visualisierung
        features_df = pd.DataFrame({'Feature': self.X_train.columns, 'Importance': importances})

        # Sortieren der Features nach ihrer Wichtigkeit
        features_df = features_df.sort_values(by='Importance', ascending=False)

        return features_df.head(number)


    def get_selected_features(self):
        pass

    def get_nonselected_features(self):
        pass
