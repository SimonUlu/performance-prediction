from src.helpers.extractors.stat_extractor.formatters.contracts.formatter import Formatter
import os

class MultiDimFormatter(Formatter):
    def __init__(self):
        pass

    def format(self):
        pass

    # def read_and_prepare(self, df, file):
    #     df = df.drop(columns =[df.columns[0]])

    #     folders = file.split(os.path.sep)

    #     if len(folders) >= 3:
    #         path_part = f"{folders[-3]}_{folders[-2]}"
    #     else: 
    #         path_part = "Fehler"

    #     new_column_names = []

    #     for column in df.columns:
    #         service_name = ''.join([i for i in column.split('-')[0] if not i.isdigit()])
            
    #         ## add path to column_name
    #         new_column_name = f"{path_part}_{service_name}" if path_part != "Fehler" else service_name
    #         new_column_names.append(new_column_name)

    #     df.columns = new_column_names

    #     return df
    
    def read_and_prepare(self, df, file):
        df = df.drop(columns =[df.columns[0]])

        folders = file.split(os.path.sep)

        if len(folders) >= 3:
            if(folders[-2] == "pod_restart"):
                path_part = "pod-restart-count"
            else:
                path_part = f"{folders[-3]}_{folders[-2]}"
        else: 
            path_part = "Fehler"

        new_column_names = []


        for index, column in enumerate(df.columns):
            service_name = path_part + "-pod-" + str(index+1)
            
            new_column_name = service_name
            new_column_names.append(new_column_name)

        df.columns = new_column_names

        return df
