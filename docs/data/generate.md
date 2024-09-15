# This is a quick overview on how to merge the different timeseries

## Preparation

- Run locust load generator script in webview under [url](http://34.68.77.173/) ## url will only work while cluster is still online. 
- Copy whole dashboard from grafana after logging in to your account (dashboard has been introduced under [grafana-setup-doc][https://github.com/SimonUlu/thesis-boutique-shop/blob/main/custom-docs/system/monitoring/grafana/grafana-extracting.md])


## Extract locust stats (f.e. requests, response_times, error_rates, etc.)

Logic can be found under: [Script](https://github.com/SimonUlu/performance-prediction/blob/main/src/helpers/extract_logs.py)

Data Preview:

```sh
Timestamp,Requests je Sekunde,Durchschnittliche Antwortzeitintervalle, ...
22,2,94.0, ...
24,3,142.33333333333334, ...
26,4,132.0, ...
28,3,119.0, ...
30,3,115.0, ...
32,5,107.0, ...
34,8,150.5, ...
36,7,193.0, ...
38,6,120.5, ...
40,13,270.9230769230769, ...
42,11,128.63636363636363, ...
```

Run: 
```sh
python3 -m src.helpers.extract_logs 
```

## Extract grafana stats (f.e. cpu_stats, memory, network_traffic, etc.)

Logic can be found under: [Script](https://github.com/SimonUlu/performance-prediction/blob/main/src/helpers/extract_stats.py)

Data Preview:

```sh
i_o_read,i_o_write,memory,network_outgoing_pod_adservice,network_outgoing_pod_cartservice, ...
0,400,12764622848,215,498,526,910,343,8618,5.47,641.0,2703.0,176,2175,594,152,320,17877,0.00236,0.00578,0.000881,0.0153,0.00197,0.00482,0.00572,0.00585,0.00917,0.00155,0.00206,0.00358,0.00226,0.000827,0.00518,0
0,403,12751368192,209,498,521,921,343,8333,51.0,641.0,2703.0,176,2295,594,148,320,17753,0.00236,0.00578,0.000876,0.0148,0.00197,0.00459,0.00629,0.00585,0.00917,0.00155,0.00206,0.00358,0.00226,0.000827,0.00516,0
0,403,12748492800,209,498,521,921,343,8333,55.3,641.0,2703.0,176,2295,594,148,320,17757,0.0023,0.00568,0.000876,0.0147,0.00195,0.00459,0.00686,0.00585,0.00917,0.00149,0.00206,0.00358,0.00226,0.000837,0.00518,0
0,403,12752961536,209,482,521,921,346,8333,59.6,641.0,2703.0,177,2226,594,148,323,17684,0.00232,0.00568,0.000876,0.0147,0.00195,0.00459,0.00469,0.00585,0.00917,0.00149,0.00213,0.00355,0.00226,0.000837,0.00501,0
0,403,12758626304,209,479,521,921,346,8651,63.9,641.0,2703.0,177,2275,594,148,323,18052,0.00238,0.00581,0.000911,0.0147,0.00195,0.0048,0.00505,0.00585,0.00917,0.00149,0.00213,0.00355,0.00226,0.000837,0.00507,0
0,403,12751814656,209,495,521,921,346,8651,68.2,641.0,2703.0,177,2275,599,150,325,18080,0.00238,0.00581,0.000911,0.0147,0.00195,0.0048,0.00541,0.00585,0.00917,0.00149,0.00213,0.00355,0.00226,0.000852,0.0051,0
0,403,12746117120,209,513,521,921,351,8651,72.5,641.0,2703.0,177,2275,599,150,325,18108,0.00238,0.00581,0.000981,0.0153,0.00195,0.0048,0.00576,0.00585,0.00917,0.00149,0.00213,0.00355,0.00226,0.000852,0.00519,0
0,392,12747350016,217,513,521,921,351,8651,357.0,641.0,2703.0,177,2275,601,150,325,18402,0.00238,0.00581,0.000981,0.0153,0.00195,0.0048,0.00612,0.00585,0.00917,0.00172,0.00224,0.00355,0.00227,0.000852,0.00524,0
0,399,12685602816,217,513,521,1028,351,9426,377.0,641.0,2703.0,177,2275,621,160,329,19338,0.00238,0.00581,0.000981,0.0153,0.00195,0.0048,0.00647,0.00585,0.00917,0.00175,0.00224,0.00355,0.00228,0.000857,0.00528,0
```

Run: 
```sh
python3 -m src.helpers.extract_stats
```


## Merge both stats 

Logic can be found under: [Script](https://github.com/SimonUlu/performance-prediction/blob/main/src/helpers/merge_stats.py)

Run:
```sh
python3 -m src.helpers.merge_stats 
```

## Finally merge all files 

Class Function can be found under: [Script](https://github.com/SimonUlu/performance-prediction/blob/main/src/app/preprocessors/csv_processor.py)

```sh
csv_processor = CsvProcessor(file="file.csv")

old_folder_path = "timeseries/merged"
new_file_path = "timeseries/merged/merged.csv

csv_processor.merge_all_files(old_folder_path, new_file_path)
```
