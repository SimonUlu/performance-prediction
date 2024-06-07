import pandas as pd
import numpy as np
import dataframe_image as dfi

# Creating a fictional dataset with daily metrics of a microservice
np.random.seed(42)  # For reproducible results
data = {
    'date': pd.date_range(start='2024-01-01', periods=10, freq='D'),
    'requests': np.random.randint(50, 200, size=10),
    'cpu_utilization': np.random.uniform(50, 90, size=10),
    'response_times': np.random.randint(100, 500, size=10)
}

# Converting to a pandas DataFrame
df = pd.DataFrame(data)

# Adding Lag Features for CPU Utilization
df['cpu_utilization_lag_1'] = df['cpu_utilization'].shift(1)  # Lag of 1 day
df['cpu_utilization_lag_2'] = df['cpu_utilization'].shift(2)  # Lag of 2 days
df['cpu_utilization_lag_3'] = df['cpu_utilization'].shift(3)  # Lag of 3 days


# Save the styled DataFrame as an image
dfi.export(df, 'dataframe_styled.png')