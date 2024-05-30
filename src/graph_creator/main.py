import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

print(plt.style.available)

# CSV-Daten in ein DataFrame umwandeln
data = pd.read_csv("timeseries/constant_load.csv")

plt.style.use('seaborn-v0_8-whitegrid')

# Erstelle einen Scatter-Plot mit grauen Punkten
plt.scatter(data['Durchschnittliche Antwortzeitintervalle'], data['Requests je Sekunde'], color='gray', label='data points')

# Berechne die lineare Regressionslinie
# np.polyfit gibt die Koeffizienten (Steigung und y-Achsenabschnitt) der Regressionslinie zurück
m, b = np.polyfit(data['Durchschnittliche Antwortzeitintervalle'], data['Requests je Sekunde'], 1)

# Berechne die Regressionslinie-Werte
regression_line = (m * data['Durchschnittliche Antwortzeitintervalle']) + b

# Zeichne die Regressionslinie
plt.plot(data['Durchschnittliche Antwortzeitintervalle'], regression_line, color='red', label='regression line')

# Achsen und Titel hinzufügen
plt.title('Korrelation zwischen Requests/Sekunde und Antwortzeit')
plt.xlabel('Durchschnittliche Antwortzeitintervalle')
plt.ylabel('Requests je Sekunde')

# Legende hinzufügen
plt.legend()

# Plot anzeigen
plt.show()