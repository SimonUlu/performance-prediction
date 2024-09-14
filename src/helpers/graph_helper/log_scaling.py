import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import StringIO
from matplotlib.ticker import MaxNLocator

data = """Time,default
2024-05-29 14:02:02,2608
2024-05-29 14:02:04,2844
2024-05-29 14:02:06,3018
2024-05-29 14:02:08,3077
2024-05-29 14:02:10,3159
2024-05-29 14:02:12,3165
2024-05-29 14:02:14,3272
2024-05-29 14:02:16,3291
2024-05-29 14:02:18,4030
2024-05-29 14:02:20,7612
2024-05-29 14:02:22,7649
2024-05-29 14:02:24,7674
2024-05-29 14:02:26,8866
2024-05-29 14:02:28,9007
2024-05-29 14:02:30,9077
2024-05-29 14:02:32,9161
2024-05-29 14:02:34,9293
2024-05-29 14:02:36,16525
2024-05-29 14:02:38,19246
2024-05-29 14:02:40,19399
2024-05-29 14:02:42,19968
2024-05-29 14:02:44,20528
2024-05-29 14:02:46,21242
2024-05-29 14:02:48,21524
2024-05-29 14:02:50,23387
2024-05-29 14:02:52,23576
2024-05-29 14:02:54,23546
2024-05-29 14:02:56,35642
2024-05-29 14:02:58,36026
2024-05-29 14:03:00,37238
2024-05-29 14:03:02,37992
2024-05-29 14:03:04,38530
2024-05-29 14:03:06,39108
2024-05-29 14:03:08,42697
2024-05-29 14:03:10,43018
2024-05-29 14:03:12,43373
2024-05-29 14:03:14,60526
2024-05-29 14:03:16,61036
2024-05-29 14:03:18,62248
2024-05-29 14:03:20,62598
2024-05-29 14:03:22,67797
2024-05-29 14:03:24,68757
2024-05-29 14:03:26,69640
2024-05-29 14:03:28,70295
2024-05-29 14:03:30,88032
2024-05-29 14:03:32,88785
2024-05-29 14:03:34,90277
2024-05-29 14:03:36,90629
2024-05-29 14:03:38,92032
2024-05-29 14:03:40,98638
2024-05-29 14:03:42,99569
2024-05-29 14:03:44,100591
2024-05-29 14:03:46,101927
2024-05-29 14:03:48,100303
2024-05-29 14:03:50,127154
2024-05-29 14:03:52,128082
2024-05-29 14:03:54,133542
2024-05-29 14:03:56,134861
2024-05-29 14:03:58,137808
2024-05-29 14:04:00,138752
2024-05-29 14:04:02,140139
2024-05-29 14:04:04,141124
2024-05-29 14:04:06,143451
2024-05-29 14:04:08,171715
2024-05-29 14:04:10,173500
2024-05-29 14:04:12,174715
2024-05-29 14:04:14,175911
2024-05-29 14:04:16,178220
2024-05-29 14:04:18,180412
2024-05-29 14:04:20,182935
2024-05-29 14:04:22,189194
2024-05-29 14:04:24,190592
2024-05-29 14:04:26,217815"""

# Daten in einen Pandas DataFrame umwandeln
df = pd.read_csv(StringIO(data), sep=",", parse_dates=['Time'], index_col='Time')

# Logarithmische Transformation der 'default'-Spalte
df['log_default'] = np.log1p(df['default'])

# Erstellen der Plots
fig, ax = plt.subplots(1, 2, figsize=(12, 6))

# Originaldaten
ax[0].plot(df.index, df['default'], marker='o', linestyle='-')
ax[0].set_title('Original Data')
ax[0].set_xlabel('Time')
ax[0].set_ylabel('Network transfer (outgoing)')
# Anpassen der x-Achse, um nur jeden 20. Zeitstempel anzuzeigen
ax[0].xaxis.set_major_locator(MaxNLocator(nbins=6))

# Logarithmisch transformierte Daten
ax[1].plot(df.index, df['log_default'], marker='o', linestyle='-', color='orange')
ax[1].set_title('Log Scaled Data')
ax[1].set_xlabel('Time')
ax[1].set_ylabel('Log(Network transfer (outgoing) + 1)')
# Anpassen der x-Achse, um nur jeden 20. Zeitstempel anzuzeigen
ax[1].xaxis.set_major_locator(MaxNLocator(nbins=6))

# Anpassung der Layouts, um Überlappungen zu vermeiden
plt.tight_layout()

plt.savefig("rolling_average.pdf", format='pdf')
# Anzeigen der Plots
plt.show()