# This is a quick overview on all preprocessing steps that have to be performed on the data

## 1. Merge timeseries of all scenarios

As we had many different scenarios set up in the beginning we began by checking if all were compatible to merge 

Code for this step can be found under [code](https://github.com/nadlig123/performance-prediction/blob/main/src/app/preprocessors/csv_processor.py)

## 2. Preprocess the data

Then the real preprocessing had to be performed. According to our thesis we have identified the following steps to be critical to our prediction. Every single step will be introduced in the following 

### a. Adding Lags

![Lags](images/lags.png)


### b. Adding Rolling Averages



### c. Adding sums over a certain timespan


### Log Scaling


### Normalization


### Standardization


### Feature Selection