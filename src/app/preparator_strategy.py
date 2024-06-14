from src.app.preprocessors.data_preparator import DataPreparator

def main():

    input_file_path = "timeseries/merged/constant-load/long/file.csv"

    data_preparator = DataPreparator(input_file_path)

    data_preparator.add_lags(columns = ['cpu_system', 'Requests je Sekunde'], num_past_timestamps=5)

    data_preparator.add_rolling_average(columns = ['cpu_system', 'long_memory'], window_size=5)

    data_preparator.data.head(20)

if __name__ == '__main__':
    main()