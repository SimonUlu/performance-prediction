import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# Erstellen eines erweiterten Beispieldatensatzes mit mehr Features
np.random.seed(0)  # Für reproduzierbare Ergebnisse
data = np.random.rand(100, 3) * [100, 1, 10]  # 100 Zeilen, 3 Features

# Normalisierung des Datensatzes
scaler_min_max = MinMaxScaler()
data_normalized = scaler_min_max.fit_transform(data)

# Standardisierung des Datensatzes
scaler_std = StandardScaler()
data_standardized = scaler_std.fit_transform(data)

# Erstellen von Subplots für den ursprünglichen, den normalisierten und den standardisierten Datensatz
fig, axs = plt.subplots(1, 3, figsize=(18, 6))  # 1 Zeile, 3 Spalten

# Plot für den ursprünglichen Datensatz
axs[0].scatter(data[:, 0], data[:, 1], color='blue')
axs[0].set_title('Ursprüngliche Daten')
axs[0].set_xlabel('Response times (in miliseconds)')
axs[0].set_ylabel('CPU-Utilization')
axs[0].set_xlim(-10, 110)  # Erweiterte x-Achse
axs[0].set_ylim(-0.5, 1.5)  # Erweiterte y-Achse

# Plot für den normalisierten Datensatz
axs[1].scatter(data_normalized[:, 0], data_normalized[:, 1], color='red')
axs[1].set_title('Normalized data')
axs[2].set_xlabel('Response times (in miliseconds)')
axs[1].set_ylabel('CPU-Utilization')
axs[1].set_xlim(-0.1, 1.1)  # Erweiterte x-Achse
axs[1].set_ylim(-0.1, 1.1)  # Erweiterte y-Achse

# Plot für den standardisierten Datensatz
axs[2].scatter(data_standardized[:, 0], data_standardized[:, 1], color='green')
axs[2].set_title('Standardized data')
axs[2].set_xlabel('Response times (in miliseconds)')
axs[2].set_ylabel('CPU-Utilization')
axs[2].set_xlim(-3, 3)  # Erweiterte x-Achse
axs[2].set_ylim(-3, 3)  # Erweiterte y-Achse

# Anzeigen der Plots
plt.tight_layout()  # Verbessert die Anordnung der Subplots
plt.show()