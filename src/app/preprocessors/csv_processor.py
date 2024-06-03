import pandas as pd

class CsvProcessor:


    def __init__(self, file):
        self.file = file

    # function to clean csv files for unneccessary columns based on base case csv
    def clean_csv(self, input_file, to_be_cleanded_file):
        df_base = pd.read_csv(input_file)

        columns = df_base.columns.tolist()

        df_to_be_cleaned = pd.read_csv(to_be_cleanded_file)

        df_cleaned = df_to_be_cleaned.reindex(columns = columns)

        return df_cleaned
    

    def rm_columns(self, string_to_be_removed):
        df = pd.read_csv(self.filepath)

        columns_to_remove = [spalte for spalte in df.columns if string_to_be_removed in spalte]

        df_cleaned = df.drop(columns_to_remove, axis=1)

        return df_cleaned
    

    def check_similarity(self, file_one, file_two):

        df1 = pd.read_csv(file_one)
        df2 = pd.read_csv(file_two)

        # extract column names 
        columns_df1 = df1.columns.tolist()
        columns_df2 = df2.columns.tolist()

        max_len = max(len(columns_df1), len(columns_df2))

        print("Compare columns:")
        print(f"{'CSV 1':<50} | {'CSV 2':<50}")
        print("-" * 105)

        for i in range(max_len):
            spalte1 = columns_df1[i] if i < len(columns_df1) else ""
            spalte2 = columns_df2[i] if i < len(columns_df2) else ""
            if spalte1 != spalte2:  # show differences
                print(f"{spalte1:<50} | {spalte2:<50}")

        # check for extra columns
        if len(columns_df2) > len(columns_df1):
            print("\nAdditional Columns:")
            for i in range(len(columns_df1), len(columns_df2)):
                print(f"- {columns_df2[i]}")

