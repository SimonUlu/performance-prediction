from src.app.preprocessors.data_preparator import DataPreparator

def main():

    input_file_path = "tea-store/five"

    data_preparator = DataPreparator(input_file_path)

    data_preparator.add_lags(columns = ['cpu_system', 'Requests je Sekunde', 'memory'], num_past_timestamps=5)

    ## add rolling averagees for cpu system, memory and frontend pod and adservice pod
    data_preparator.add_rolling_average(columns = ['Requests je Sekunde','cpu_system', 'memory', 'cpu_pod-pod-1', 'cpu_pod-pod-11', 'cpu_pod-pod-8'], window_size=5)

    data_preparator.add_cumulative_sum(['cpu_system', 'memory'])

    data_preparator.log_scale(['network_outgoing_system', 'memory', 'cpu_system'])

    data_preparator.save_prepared_data()

if __name__ == '__main__':
    main()