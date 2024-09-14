import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# Erstellen eines erweiterten Beispieldatensatzes mit mehr Features
np.random.seed(0)  # Für reproduzierbare Ergebnisse
data = np.random.rand(100, 3) * [100, 4, 10]  # 100 Zeilen, 3 Features

# Isolieren der Features für die Visualisierung
response_times = data[:, 0]  # Response times (x-Achse)
cpu_utilization = data[:, 1].reshape(-1, 1)  # CPU-Utilization (y-Achse), Reshape für die Skalierer

# Normalisierung des CPU-Utilization Features
scaler_min_max = MinMaxScaler()
cpu_utilization_normalized = scaler_min_max.fit_transform(cpu_utilization)

# Standardisierung des CPU-Utilization Features
scaler_std = StandardScaler()
cpu_utilization_standardized = scaler_std.fit_transform(cpu_utilization)

# Erstellen von Subplots für den ursprünglichen, den normalisierten und den standardisierten Datensatz
fig, axs = plt.subplots(1, 3, figsize=(18, 6))  # 1 Zeile, 3 Spalten

# Plot für den ursprünglichen Datensatz
axs[0].scatter(response_times, cpu_utilization, color='blue')
axs[0].set_title('Original data')
axs[0].set_xlabel('Response times (in milliseconds)')
axs[0].set_ylabel('CPU-Utilization')
axs[0].set_xlim(-10, 110)  # Erweiterte x-Achse
axs[0].set_ylim(-0.5, 4)  # Erweiterte y-Achse

# Plot für den normalisierten Datensatz
axs[1].scatter(response_times, cpu_utilization_normalized, color='red')
axs[1].set_title('Normalized data')
axs[1].set_xlabel('Response times (in milliseconds)')
axs[1].set_ylabel('CPU-Utilization')
axs[1].set_xlim(-10, 110)  # Gleiche x-Achse wie Original
axs[1].set_ylim(-0.1, 1.1)  # Erweiterte y-Achse für Normalisierung

# Plot für den standardisierten Datensatz
axs[2].scatter(response_times, cpu_utilization_standardized, color='green')
axs[2].set_title('Standardized data')
axs[2].set_xlabel('Response times (in milliseconds)')
axs[2].set_ylabel('CPU-Utilization')
axs[2].set_xlim(-10, 110)  # Gleiche x-Achse wie Original
axs[2].set_ylim(-3, 3)  # Erweiterte y-Achse für Standardisierung

# Anzeigen der Plots
plt.tight_layout()  # Verbessert die Anordnung der Subplots
plt.savefig("stand_norm.pdf", format='pdf')
plt.show()