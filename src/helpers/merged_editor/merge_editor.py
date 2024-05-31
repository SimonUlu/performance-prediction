import pandas as pd

class MergeEditor:

    def __init__(self):
        pass

    def convert_pod_restarts(self, input_file_path, output_file_path):
        # Laden der CSV-Datei
        df = pd.read_csv(input_file_path)

        # Entfernen der Spalten, die 'pod_restart' enthalten
        columns_to_drop = [col for col in df.columns if 'pod_restart' in col]
        df.drop(columns=columns_to_drop, axis=1, inplace=True)

        # Hinzufügen der neuen Spalten für Pod-Restarts
        new_columns = [
            "pod_restart_{container=\"main\", instance=\"kube\"",
            "pod_restart_{container=\"main\", instance=\"kube.1\"",
            "pod_restart_{container=\"redis\", instance=\"kube\"",
            "pod_restart_{container=\"server\", instance=\"kube\"",
            "pod_restart_{container=\"server\", instance=\"kube.1\"",
            "pod_restart_{container=\"server\", instance=\"kube.2\"",
            "pod_restart_{container=\"server\", instance=\"kube.3\"",
            "pod_restart_{container=\"server\", instance=\"kube.4\"",
            "pod_restart_{container=\"server\", instance=\"kube.5\"",
            "pod_restart_{container=\"server\", instance=\"kube.6\"",
            "pod_restart_{container=\"server\", instance=\"kube.7\"",
            "pod_restart_{container=\"server\", instance=\"kube.8\"",
            "pod_restart_{container=\"server\", instance=\"kube.9\""
        ]

        for column in new_columns:
            df[column] = 0  # Alle neuen Spalten mit 0 initialisieren

        # Speichern der veränderten DataFrame in einer neuen CSV-Datei
        df.to_csv(output_file_path, index=False)

        print("Die CSV-Datei wurde erfolgreich modifiziert und gespeichert.")

    def convert_extra_pods():
        pass


# Beispiel für die Verwendung der Klasse
converter = MergeEditor()
base_path = "constant-load/medium"
input_file_path = "timeseries/merged/" + base_path + "/file.csv"
output_file_path = "timeseries/merged/" + base_path + "/new.csv"
converter.convert_pod_restarts(input_file_path, output_file_path)