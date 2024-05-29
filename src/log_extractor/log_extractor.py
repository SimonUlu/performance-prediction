import pandas as pd
import re
import os
print("Aktuelles Arbeitsverzeichnis:", os.getcwd())

filepath = "locust_logs/meine_logs.txt"

class LogExtractor:

    def __init__(self, filepath):
        self.filepath = filepath

    def process_log_file(self, time_interval):
        time_spent_in_seconds = 0
        current_users_count = 0
        timestamps = []
        requests = []
        failures = []
        avg_response_times = []
        current_users = []

        with open(self.filepath, "r") as file:
            for line in file:
                line = line.strip()
                if line.startswith("Aggregated"):
                    time_spent_in_seconds += time_interval
                    current_users_count += time_interval

                    line = line.replace("|", "")
                    line = re.sub(r'\s+', '|', line)
                    line = line.replace(" ", "")
                    data = line.split("|")

                    timestamps.append(time_spent_in_seconds)
                    requests.append(data[1])
                    failures.append(data[2])
                    avg_response_times.append(data[3])
                    current_users.append(current_users_count)

            requests = [int(r) for r in requests]
            avg_response_times = [int(t) for t in avg_response_times]
            requests_diff = [j-i for i, j in zip(requests[:-1], requests[1:])]
            avg_response_time_intervals = self.__calculate_avg_response_time_intervals(requests, avg_response_times, requests_diff)

            return avg_response_time_intervals, requests, timestamps
    

    
    def __calculate_avg_response_time_intervals(self, requests, avg_response_times, requests_diff):
        avg_response_time_intervals = []
        for i in range(1, len(requests)):
            total_response_time_current = avg_response_times[i] * requests[i]
            total_response_time_previous = avg_response_times[i-1] * requests[i-1]
            total_response_time_interval = total_response_time_current - total_response_time_previous
            requests_interval = requests_diff[i-1]

            if requests_interval > 0:
                avg_response_time_interval = total_response_time_interval / requests_interval
                avg_response_time_interval = max(avg_response_time_interval, 0)
            else:
                avg_response_time_interval = 0

            avg_response_time_intervals.append(avg_response_time_interval)

        return avg_response_time_intervals

    # To-do save stats into stats folder
    def __save_stats(self):
        pass

