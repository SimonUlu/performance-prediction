import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# Daten generieren
np.random.seed(42)
data = {
    'date': pd.date_range(start='2024-01-01', periods=10, freq='D'),
    'requests': np.random.randint(50, 200, size=10),
    'cpu_utilization': np.random.uniform(50, 90, size=10),
    'response_times': np.random.randint(100, 500, size=10)
}
df = pd.DataFrame(data)
df['cpu_utilization_lag_1'] = df['cpu_utilization'].shift(1)
df['cpu_utilization_lag_2'] = df['cpu_utilization'].shift(2)
df['cpu_utilization_lag_3'] = df['cpu_utilization'].shift(3)

# DataFrame als PDF speichern
with PdfPages('dataframe_export.pdf') as pdf:
    fig, ax = plt.subplots(figsize=(12, 6))  # Größere Tabelle für PDF
    ax.axis('tight')
    ax.axis('off')
    table = ax.table(cellText=df.values, colLabels=df.columns, cellLoc='center', loc='center')

    pdf.savefig(fig, bbox_inches='tight')  # Speichern als PDF
    plt.close()

print("PDF wurde erstellt: dataframe_export.pdf")
