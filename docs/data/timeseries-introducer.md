# This doc explains the different timeseries data that can be found under folder /timeseries and explains how one can create the final data

## 1. Grafana

This folder contains only the grafana stats that are imported via csv-files merged to one big timeseries


## 2. Locust

After extracting logs via script extract_logs from raw txt file these files are being created. Also after the processing they are in csv-format over the same timespan as grafana files

## 3. Merged 

This folder contains the previously created (and separated timeseries) locust and grafana merged together but still for every scenario separated

This is especially important for the final evaluation as this is a nice base for comparing the final results of our models before and after preprocessing

## 4. Preprocessed final

This folder contains the separated (for each scenario) files after executing the data preparator strategy introduced in [file](/src/app/preparator_strategy.py)

## 5. Combined

This folder contains the final preprocessed data in csv-format with all scenarios merged together (as one single big file)