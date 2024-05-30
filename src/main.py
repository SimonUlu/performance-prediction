from log_extractor.log_extractor import LogExtractor

def main():
    input_filepath = "assets/locust_logs/constant-load/long/constant_lng.txt"
    output_filepath = "timeseries/constant_load.csv"

    log_extractor = LogExtractor(input_filepath)
    avg_response_time_intervals, requests, timestamps = log_extractor.process_log_file(2, output_filepath)
    print("Durchschnittliche Antwortzeitintervalle:", avg_response_time_intervals)
    print("Timestamps:", timestamps)
    print("Requests je Sekunde", requests)


if __name__ == "__main__":
    main()

