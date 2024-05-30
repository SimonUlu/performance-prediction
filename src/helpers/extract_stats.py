from src.helpers.extractors.stat_extractor.formatters.factories.formatter_factory import FormatterFactory
import pandas as pd
import os
from src.helpers.extractors.stat_extractor.formatters.single_dim_formatter import SingleDimFormatter
from src.helpers.extractors.stat_extractor.formatters.multi_dim_formatter import MultiDimFormatter
from src.helpers.extractors.stat_extractor.formatters.stat_merger import StatMerger

def main():

    # use formatter Factory
    formatter_factory = FormatterFactory()
    stat_merger = StatMerger("timeseries/merged/new.csv")

    # set base_path to scenario you want to get the stats from 
    base_path = "assets/grafana_stats/constant_load/long"

    for root, dirs, files in os.walk(base_path):
        for file in files:
            file_path = os.path.join(root, file)
            df = formatter_factory.prepare_data(file_path)
            stat_merger.add_dataframe(df)

    print(stat_merger.dataframes)   
    stat_merger.create_time_series()


if __name__ == "__main__":
    main()
