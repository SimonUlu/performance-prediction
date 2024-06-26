from src.helpers.extractors.stat_extractor.formatters.factories.formatter_factory import FormatterFactory
import pandas as pd
import os
from src.helpers.extractors.stat_extractor.formatters.stat_merger import StatMerger

def main():

    custom_path = "tea-store/five"

    # use formatter Factory
    formatter_factory = FormatterFactory()

    ##set output file path
    stat_merger = StatMerger("timeseries/grafana/" + custom_path + "/file.csv")

    # set input file path
    base_path = "assets/grafana_stats/" + custom_path

    for root, dirs, files in os.walk(base_path):
        for file in files:
            file_path = os.path.join(root, file)
            df = formatter_factory.prepare_data(file_path)
            stat_merger.add_dataframe(df)
 
    stat_merger.create_time_series()


if __name__ == "__main__":
    main()
