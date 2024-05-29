from log_extractor.log_extractor import LogExtractor

def main():
    filepath = "locust_logs/meine_logs.txt"
    log_extractor = LogExtractor(filepath)
    avg_response_time_intervals, requests, timestamps = log_extractor.process_log_file(2)
    print("Durchschnittliche Antwortzeitintervalle:", avg_response_time_intervals)
    print("Timestamps:", timestamps)
    print("Requests je Sekunde", requests)


if __name__ == "__main__":
    main()

