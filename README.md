# 1. Get Started

## a. create a virtual environment

```sh
 python3 -m venv venv
```

## b. activate the venv

```sh
source venv/bin/activate
```

## c. install packages

```sh
pip install -r requirements.txt
```


# 2. Extract all needed stats

For the data extraction please refer to the docs under [docs](/docs/data/generate.md). This guide will help you understand the extraction of the locust stats (such as requests, response times) and the extraction of the stats retrieved from grafana (timeseries of cpu-util and so on). 

Furthermore it guides you how to merge these stats, to finally get one timeseries with all merged stats.

To better understand the different timeseries that are being created by the various scripts check [doc](/docs/data/timeseries-introducer.md)


# 3. Data Preprocessing

As previously discussed we generated most of the data from prometheus and grafana. The following pseudocode gives a quick overview on how we prepared the data for prediciction:


```sh
data_preparator = DataPreparator(input_file_path)

data_preparator.add_lags(columns = ['cpu_system', 'Requests je Sekunde', 'memory'], num_past_timestamps=5)

## add rolling averagees for cpu system, memory and frontend pod and adservice pod
data_preparator.add_rolling_average(columns = ['Requests je Sekunde','cpu_system', 'memory', 'cpu_pod-pod-1', 'cpu_pod-pod-11', 'cpu_pod-pod-8'], window_size=5)

data_preparator.add_cumulative_sum(['cpu_system', 'memory'])

data_preparator.log_scale(['network_outgoing_system', 'memory', 'cpu_system'])

data_preparator.normalize(['memory', ...])

data_preparator.save_prepared_data()
```

For further understanding please refer to the [subdocumentation](/docs/data/preprocess.md)

# 4. Feature Selection

The feature selection consists of two main parts.  
## a. Remove unneccessary features from our data

At first we tried to identify features that weaken our prediction results by introducing unneccessary noise. This is especially important as for future research in this field it is very interesting to see which performance measures are especially important for predicting response times.

Please refer to [subdocumentation](/docs/data/feature_selection.md)

## b. Find the most important features

The second part consists of finding the most important features. The most important features are being used for adding lags, etc.

Please refer to [subdocumentation](/docs/data/feature_selection.md)

# 5. Structure of the repository

### 1. Assets-folder

Contains the following:
- Extracted stats from grafana/prometheus from all runs from all reference applications
- Extracted stats from locust runs from all runs from all reference applications
- images produced for the thesis

### 2. Docs

Only contains subdocumentations for greater understanding of the code

### 3. TimeSeries

Contains the before merged and preprocessed timeseries files that were extracted from both grafana and locust 

### 4. Src 

This is the main folder for the project where all functionality has been declared. For further information on the structure for this repo check out, following docs:
[Structure Documentation](./docs/structure/ProjectStructure.md)
