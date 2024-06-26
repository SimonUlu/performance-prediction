from src.helpers.extractors.log_extractor.log_extractor import LogExtractor

def main():

    custom_path = "tea-store/five"

    # set filepath to be used
    input_filepath = "assets/locust_logs/"+ custom_path +"/file.txt"

    # set new filepath where files will be saved
    output_filepath = "timeseries/locust/" + custom_path + "/file.csv"

    log_extractor = LogExtractor(input_filepath)
    avg_response_time_intervals, requests, timestamps = log_extractor.process_log_file(2, output_filepath)
    print("Durchschnittliche Antwortzeitintervalle:", avg_response_time_intervals)
    print("Timestamps:", timestamps)
    print("Requests je Sekunde", requests)


if __name__ == "__main__":
    main()

