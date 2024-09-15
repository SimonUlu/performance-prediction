# In this docs you can see how the features got prepared and potentially selected

please refer to [notebook](../../src/dev/feature_engineering.ipynb)

## 1. Following strategies have been conducted to deal with nan-values as many modells cant deal with team

### a. Impute to median

```sh
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy='mean')
```

### b. Impute to mean


```sh
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy='median')
```


## 2. Select features that should not be conducted any more

- the implemented code for this task can be found under [code](/src/app/feature_selection/recursive_feature_selector.py)

use a base model that has been validated with the dataset to at least deliver okay results. and then use it recursively to find features that don't contribute to the models performance at all to eliminate the unneccessary noise from the dataset

```sh
estimator = self.base_model

# Feature Selection
selector = RFECV(estimator, step=step, cv=cross_validation_folds)
selector = selector.fit(self.X_train, self.y_train)

print("Optimale Anzahl von Features : %d" % selector.n_features_)

selected_features = self.X_train.columns[selector.support_]
print("Ausgewählte Felder", selected_features)

not_selected_features = self.X_train.columns[~selector.support_]
print("Nicht ausgewählte Features:", not_selected_features)

return not_selected_features
```


## 3. Select most important features to be used for adding lags and rolling averages

- the implemented code can also be found under [code](/src/app/feature_selection/recursive_feature_selector.py)

This task is to find the most important features for our analysis to have a basic understanding of the dataset and to be able to add lags or rolling averages most efficiently

```sh
# Angenommen, X_train und y_train sind bereits definiert und vorbereitet
# Trainieren des Modells
model = DecisionTreeRegressor(random_state=42)
model.fit(X_train, y_train)

# Holen der Feature Importances
importances = model.feature_importances_

# Konvertieren der Importances in einen DataFrame für eine bessere Visualisierung
features_df = pd.DataFrame({'Feature': X_train.columns, 'Importance': importances})

# Sortieren der Features nach ihrer Wichtigkeit
features_df = features_df.sort_values(by='Importance', ascending=False)

# Anzeigen der Top 3-5 Features
print(features_df.head(20))
```

