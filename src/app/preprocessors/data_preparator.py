import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler

class DataPreparator:

    def __init__(self, file_path):
        self.file_path = file_path
        self._data = pd.read_csv(file_path)

    # function to add lags to columns specified in header and time specified in heder
    def add_lags(self, columns, num_past_timestamps):
        for column in columns:
            for lag in range(1, num_past_timestamps + 1):
                self.data[f'{column}_lag_{lag}'] = self.data[column].shift(lag)

    # function to add rolling averages -> over a spec period of time
    def add_rolling_average(self, columns, window_size):
        for column in columns:
            self.data[f'{column}_rolling_avg_{window_size}'] = self.data[column].rolling(window=window_size).mean()
        
    def add_cumulative_sum(self, columns):
        for column in columns:
            self.data[f'{column}_cumsum'] = self.data[column].cumsum()
        
    def log_scale(self, columns):
        for column in columns:
            self.data[f'{column}_log_scaled'] = np.log1p(self.data[column])
        
    def normalize(self, columns):
        scaler = MinMaxScaler()
        for column in columns:
            self.data[f'{column}_normalized'] = scaler.fit_transform(self.data[[column]])
        
    def standardize(self, columns):
        scaler = StandardScaler()
        for column in columns:
            self.data[f'{column}_standardized'] = scaler.fit_transform(self.data[[column]])
        
    def save_prepared_data(self, output_file_path):
        self.data.to_csv(output_file_path, index=False)

    @property
    def data(self):
        return self._data
    
    @data.setter
    def data(self, value):
        self._data = value


    def remove_nan_values(self):
        pass


    