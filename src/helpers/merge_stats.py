import pandas as pd

base_path = "tea-store/five"

first_path = "timeseries/grafana/" + base_path + "/file.csv"
second_path = "timeseries/locust/"  + base_path + "/file.csv"

new_path = "timeseries/merged/" + base_path + "/file.csv"

df1 = pd.read_csv(second_path)
df2 = pd.read_csv(first_path)

merged_df = pd.concat([df1, df2], axis=1)

merged_df.reset_index(drop=True, inplace=True)

print(merged_df.head(5))

merged_df.to_csv(new_path, index=False)
