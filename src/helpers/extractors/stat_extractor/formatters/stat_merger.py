import pandas as pd

class StatMerger:

    def __init__(self, path):
        self.path = path
        self.dataframes = []

    
    def add_dataframe(self, df):
        self.dataframes.append(df)

    def create_time_series(self):
        """Führt alle DataFrames zusammen und speichert das Ergebnis als CSV."""
        if not self.dataframes:
            print("Keine DataFrames zum Zusammenführen vorhanden.")
            return
        
        # Zusammenführen der DataFrames
        merged_df = pd.concat(self.dataframes, axis=1)
        
        # Speichern als CSV
        merged_df.to_csv(self.path, index=False)
        print(f"CSV erfolgreich gespeichert unter: {self.path}")