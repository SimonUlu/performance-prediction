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