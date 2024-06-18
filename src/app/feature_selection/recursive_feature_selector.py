from sklearn.feature_selection import RFECV
import pandas as pd
from sklearn.model_selection import train_test_split

class RecursiveFeatureSelector:

    def __init__(self, base_model, input_file_path, y_cols, to_be_removed_cols, ):
        self.base_model = base_model
        self.df = pd.read_csv(input_file_path)
        self.X = self.df.drop(['Durchschnittliche Antwortzeitintervalle', 'Requests je Sekunde', 'Timestamp'], axis=1)
        self.y = self.df['Durchschnittliche Antwortzeitintervalle']
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=42)


    def select_features(self, step, cross_validation_folds):
        estimator = self.base_model

        # Feature Selection
        selector = RFECV(estimator, step=step, cv=cross_validation_folds)
        selector = selector.fit(self.X_train, self.y_train)

        selected_features = self.X_train.columns[selector.support_]

        return selected_features




    def get_selected_features(self):
        pass

    def get_nonselected_features(self):
        pass
