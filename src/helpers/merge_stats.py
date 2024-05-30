import pandas as pd

first_path = "timeseries/grafana/new.csv"
second_path = "timeseries/locust/constant_load.csv"

df1 = pd.read_csv(second_path)
df2 = pd.read_csv(first_path)

merged_df = pd.concat([df1, df2], axis=1)

merged_df.reset_index(drop=True, inplace=True)

print(merged_df.head(5))

merged_df.to_csv("timeseries/merged/new.csv", index=False)
