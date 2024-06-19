from src.app.preprocessors.data_preparator import DataPreparator

def main():

    input_file_path = "timeseries/merged/constant-load/long"

    data_preparator = DataPreparator(input_file_path)

    data_preparator.add_lags(columns = ['cpu_system', 'Requests je Sekunde'], num_past_timestamps=5)

    ## add rolling averagees for cpu system, memory and frontend pod and adservice pod
    data_preparator.add_rolling_average(columns = ['cpu_system', 'memory', 'cpu_pod-pod-1', 'cpu_pod-pod-11'], window_size=5)

    data_preparator.add_cumulative_sum(['cpu_system'])

    data_preparator.log_scale(['network_outgoing_system', 'memory', 'i_o_read', 'i_o_write'])

    print(data_preparator.data.head(20))

if __name__ == '__main__':
    main()