import pandas as pd
from sklearn.model_selection import train_test_split

class Preprocessor:
    def __init__(self, filepath, y_label, drop_columns=None, test_size=0.2, random_state=42):
        self.filepath = filepath
        self.y_label = y_label
        self.drop_columns = drop_columns if drop_columns is not None else []
        self.test_size = test_size
        self.random_state = random_state
        self.X_train = pd.DataFrame()
        self.X_test = pd.DataFrame()
        self.y_train = pd.Series() 
        self.y_test = pd.Series()  


    def preprocess(self):
        try:
            df = pd.read_csv(self.filepath)
        except Exception as e:
            raise FileNotFoundError(f"Die Datei unter {self.filepath} konnte nicht gefunden oder gelesen werden: {e}")

        if not set(self.drop_columns).issubset(df.columns):
            raise ValueError("Ein oder mehrere Spalten zum Entfernen existieren nicht im DataFrame.")

        if self.y_label not in df.columns:
            raise ValueError(f"Die Zielvariable '{self.y_label}' existiert nicht im DataFrame.")

        df = df.drop(self.drop_columns, axis=1)

        X = df.drop(self.y_label, axis=1)
        y = df[self.y_label]

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state)
        

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



