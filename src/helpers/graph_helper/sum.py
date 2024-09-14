import numpy as np
import matplotlib.pyplot as plt

# Daten generieren
# Simuliere CPU-Auslastungsdaten (in Prozent) für 100 Timestamps
cpu_utilization = np.random.rand(100) * 100  # Zufällige Werte zwischen 0 und 100

# Daten aufbereiten
# Berechne die Summe der CPU-Auslastung für jede Gruppe von 5 Timestamps
cumulative_cpu_utilization = [np.sum(cpu_utilization[i:i+5]) for i in range(0, len(cpu_utilization), 5)]

# Berechne die Durchschnittswerte der CPU-Auslastung für jede Gruppe von 5 Timestamps
average_cpu_utilization = [np.mean(cpu_utilization[i:i+5]) for i in range(0, len(cpu_utilization), 5)]

# Plot erstellen
plt.figure(figsize=(10, 6))


plt.style.use('seaborn-v0_8-whitegrid')

# Kumulierte Werte
plt.plot(cumulative_cpu_utilization, marker='o', linestyle='-', label='Sum (Cpu-Utilization)')

# Durchschnittliche (nicht kumulierte) Werte
plt.plot(average_cpu_utilization, marker='x', linestyle='--', label='Average (Cpu-Utilizazion)')

plt.title('CPU-Utilization over time')
plt.xlabel('Intervalls (25 seconds - 5 timestamps)')
plt.ylabel('CPU-Utilization')
plt.legend()
plt.grid(True)
plt.savefig("cpu_util.pdf", format='pdf')
plt.show()