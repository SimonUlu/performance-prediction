from scipy.stats import poisson

# Durchschnittliche Anfragen pro Sekunde (RPS)
average_rps = 800
# Gewünschtes Vertrauensniveau
confidence_level = 0.99999

# Berechnung des Spitzenwerts
# Wir verwenden die inverse CDF (Percent-Point-Funktion), um den Wert zu finden,
# bei dem das Vertrauensniveau erreicht wird.
peak_rps = poisson.ppf(confidence_level, average_rps)

print(f"Bei einem durchschnittlichen RPS von {average_rps} und einem Vertrauensniveau von {confidence_level*100}%,")
print(f"ist der geschätzte Spitzenwert (maximale Anzahl von Anfragen pro Sekunde): {peak_rps:.2f}")