from src.helpers.extractors.stat_extractor.formatters.contracts.formatter import Formatter
import os

class SingleDimFormatter(Formatter):
    def __init__(self):
        pass


    def read_and_prepare(self, df, file):

        df = df.iloc[:, [1]]
        folders = file.split(os.path.sep)

        if len(folders) >= 3:
            if(folders[-2] == "system"):
                new_column_name = f"{folders[-3]}_{folders[-2]}"
            else:
                new_column_name = f"{folders[-2]}"
        else:
            new_column_name = "Fehler"

        # rename column
        df.columns = [new_column_name]

        return df
    

    def format(self, data):
        pass