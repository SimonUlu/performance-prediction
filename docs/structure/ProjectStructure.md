# ProjectStructure

## 1. App Folder


### 1.1 Preprocessors
[Preprocessors](./../../src/app/preprocessors/).

This folder holds all the functionalities for preprocessing

- Preprocessing of csv files to assuring that they all match in format when they will be merged [CSV-Preprocessor](./../../src/app/preprocessors/csv_processor.py)
- Data Prepator that implements all the functions to convert normal features to f.e. log-scaled, lags, standardized format, etc. [Data-Preprocessor](./../../src/app/preprocessors/data_preparator.py)
- Feature Selector: Class that implements functions to select most important features after pca and randomized lasso [FeatureSelector](./../../src/app/preprocessors/feature_selector.py)
- Preprocessor that splits data into train and test [File](./../../src/app/preprocessors/preprocessor.py)


### 1.2 Feature Selector
[feature-selector](./../../src/app/feature_selection/)
This only holds two files that take different approaches in selecting the most important features for our dataset

### 1.3 Predictor
This folder initially holds 5 Jupyter Notebooks that train different models. I chose to use notebooks for this to be able to reproduce my results easier for later use and display. 

#### 1.3.1 Contracts

[AbstractClass](./../../src/app/predictor/contracts/model.py)

This only holds an abstract class that helps to implement all models that i want to use later, so that i dont have to implement some functions over and over

#### 1.3.2 Models

[Models](./../../src/app/predictor/evaluation_models/)
In this part the models with the chosen configurations are introduced. They all basically use the model abstract class and implement models with different configuration and different hyperparameters


#### 1.3.3 Main Notebooks

- [Evaluator](./../../src/app/predictor/evaluator.ipynb) holds all ensemble models and the linear regressor
- [SVR](./../../src/app/predictor/svr.ipynb) holds the support vector regressor
- [Tensorflow](./../../src/app/predictor/tensorflow.ipynb) holds the neural network
- [Test](./../../src/app/predictor/test.ipynb) holds only tests that were used to understand the database better



## 2. Dev
[Dev](./../../src/dev/).
This folder was basically only used to test some functions and functionalities that i later on wanted to implement. 


## 3. Helpers


### 3.1 Graph Helper

This area is rather insignificant for the reader. It only holds python files that were used to create the figures that were used in the master-thesis. For example plots, etc. 

### 3.2 Extractors

Greater goal of the extractors is to prepare data from two different sources (grafana & locust) so that they can later on be merged

#### 3.2.1 Log Extractor
[Log Extractor](./../../src/helpers/extractors/log_extractor/).
Two files
- 1 Python file with classes to extract stats from locust runs into unified csv-format to later on combine them with prometheus data
- 1 Notebook only for testing purposes of the functions and classes

#### 3.2.2 Stat Extractor
[Stat Extractor](./../../src/helpers/extractors/stat_extractor/formatters/).
- Two formatters for Single Dimensional Data and Multi Dimensional Data that are used by the script [extract_stats.py](./../../src/helpers/extract_stats.py). With implemented factories and abstract classes. The greater goal of these files is to convert the data that we retrieved from grafana to an unified format to be able to merge them with the data from the locust runs

### 3.3 Merge Editor

[File](./../../src/helpers/merged_editor/)

This only holds one python class that was used for preparing already merged files.

Input = One file that already has merged data from locust and grafana

Output = One file with new structure for the pod_restarts as i had naming conflicts with some columns 