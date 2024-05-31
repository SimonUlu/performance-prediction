import pandas as pd
from src.helpers.extractors.stat_extractor.formatters.single_dim_formatter import SingleDimFormatter
from src.helpers.extractors.stat_extractor.formatters.multi_dim_formatter import MultiDimFormatter

class FormatterFactory:
    def __init__(self):
        # Hier könnten weitere Initialisierungen stattfinden
        pass

    def get_formatter(self, df):
        if len(df.columns) > 2:
            return MultiDimFormatter()
        else:
            return SingleDimFormatter()

    def prepare_data(self, file_path):
        df = pd.read_csv(file_path)
        formatter = self.get_formatter(df)
        return formatter.read_and_prepare(df, file_path)