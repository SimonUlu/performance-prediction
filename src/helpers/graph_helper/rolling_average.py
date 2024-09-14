import pandas as pd
import matplotlib.pyplot as plt

data = """Time,default
2024-05-29 10:22:36,0.0015
2024-05-29 10:22:38,0.0020
2024-05-29 10:22:40,0.0030
2024-05-29 10:22:42,0.0045
2024-05-29 10:22:44,0.0070
2024-05-29 10:22:46,0.0050
2024-05-29 10:22:48,0.0035
2024-05-29 10:22:50,0.0025
2024-05-29 10:22:52,0.0030
2024-05-29 10:22:54,0.0035
2024-05-29 10:22:56,0.0040
2024-05-29 10:22:58,0.0060
2024-05-29 10:23:00,0.0080
2024-05-29 10:23:02,0.0070
2024-05-29 10:23:04,0.0050
2024-05-29 10:23:06,0.0040
2024-05-29 10:23:08,0.0035
2024-05-29 10:23:10,0.0030
2024-05-29 10:23:12,0.0025
2024-05-29 10:23:14,0.0040
2024-05-29 10:23:16,0.0060
2024-05-29 10:23:18,0.0090
2024-05-29 10:23:20,0.0070
2024-05-29 10:23:22,0.0050
2024-05-29 10:23:24,0.0030
2024-05-29 10:23:26,0.0020
2024-05-29 10:23:28,0.0025
2024-05-29 10:23:30,0.0030
2024-05-29 10:23:32,0.0035
2024-05-29 10:23:34,0.0050
2024-05-29 10:23:36,0.0070
2024-05-29 10:23:38,0.0050
2024-05-29 10:23:40,0.0030
2024-05-29 10:23:42,0.0020
2024-05-29 10:23:44,0.0015
2024-05-29 10:23:46,0.0025
2024-05-29 10:23:48,0.0040
2024-05-29 10:23:50,0.0060
2024-05-29 10:23:52,0.0080
2024-05-29 10:23:54,0.0050
2024-05-29 10:23:56,0.0030
2024-05-29 10:23:58,0.0020
2024-05-29 10:24:00,0.0030
2024-05-29 10:24:02,0.0040
2024-05-29 10:24:04,0.0050
2024-05-29 10:24:06,0.0070
2024-05-29 10:24:08,0.0050
2024-05-29 10:24:10,0.0030
2024-05-29 10:24:12,0.0020
2024-05-29 10:24:14,0.0015
2024-05-29 10:24:16,0.0020
2024-05-29 10:24:18,0.0035
2024-05-29 10:24:20,0.0050
2024-05-29 10:24:22,0.0070
2024-05-29 10:24:24,0.0040
2024-05-29 10:24:26,0.0025
2024-05-29 10:24:28,0.0020
2024-05-29 10:24:30,0.0030
2024-05-29 10:24:32,0.0045
2024-05-29 10:24:34,0.0060"""

# Der restliche Code bleibt gleich. Dieser angepasste Datensatz zeigt nun deutliche Schwankungen mit Peaks und Tälern,
# was den Rolling Average besser zur Geltung bringt.



# Daten in ein DataFrame umwandeln
from io import StringIO
df = pd.read_csv(StringIO(data), parse_dates=['Time'], index_col='Time')

# Berechnung des Rolling Averages mit einem Fenster von 5 Messungen
df['RollingAvg'] = df['default'].rolling(window=5).mean()

# Plot-Stil festlegen
plt.style.use('seaborn-v0_8-whitegrid')

# Erstellung des Plots
plt.figure(figsize=(10, 6))
plt.plot(df.index, df['default'], label='CPU Utilization', alpha=0.5)
plt.plot(df.index, df['RollingAvg'], label='Rolling Average', color='red', linewidth=2)
plt.title('CPU Utilization with Rolling Average')
plt.xlabel('Time')
plt.ylabel('CPU Utilization')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()

plt.savefig("rolling_average.pdf", format='pdf')
# Anzeigen des Plots
plt.show()