# This is a guide on how to individualize and optimize ml models with tensorflow and keras

## 1. Build a sequential Model

```sh
model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10)
])
```

- Define a loss function

```sh
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
```

- compile model

```sh
model.compile(optimizer='adam',
              loss=loss_fn,
              metrics=['accuracy'])
```

- Train and evaluate model on train dataset

```sh
model.fit(x_train, y_train, epochs=5)
```

- Evaluate model on test data

```sh
model.evaluate(x_test,  y_test, verbose=2)
```

### 1.1. Characteristics of a sequential model

- Dropout:

Dropout ist eine Regularisierungstechnik, die die Überanpassung (Overfitting) in neuronalen Netzen verhindert. Überanpassung tritt auf, wenn ein Modell die Trainingsdaten zu genau lernt, einschließlich des Rauschens und der Ausreißer, was zu einer schlechten Generalisierung auf neue Daten führt. Technisch gesehen wird beim Dropout während des Trainings eines neuronalen Netzes für jede Trainingsiteration ein Teil der Neuronen (samt ihren Verbindungen) zufällig ausgewählt und temporär aus dem Netzwerk entfernt. Die Wahrscheinlichkeit, mit der ein Neuron ausgeschaltet wird, wird durch den Dropout-Rate-Parameter bestimmt, der üblicherweise zwischen 0,2 und 0,5 liegt. Dies zwingt das Netzwerk dazu, redundante Repräsentationen der Daten zu lernen, da es sich nicht auf die Anwesenheit spezifischer Neuronen verlassen kann.

- Aktivierungsfunktion:

An sich gibt es zum Beispiel tangens hyperbolikus (tanh), sigmoid oder relu aktivierungsfunktionen. Bei großen Netzen wird oftmal in den fc layern, meist die relu funktion verwendet weil sie auch werte gen 0 sehr gut gewichtet und ebenfalls sehr schnell ist. 

- Layer:

Wähle eine sinnvolle Struktur. Zu viele Layer führen oft dazu dass ein Modell overfitted. Daher sollte man mehrere Architekture miteinander austesten.

- Input Layer:

Zum Flatten der Daten je nach Input, muss das daran angepasst werden. 


## Build a functional modell

### choose this if your prediction would benefit from a more complexe model structure that should also predict non linear interrelationships

- import following

```sh
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Concatenate
```

- Define input layer

```sh
inputs = Input(shape=(X_train_scaled.shape[1],))
```

- Define fc layers and etc.

could also be more:

```sh
x = Dense(64, activation='relu')(inputs)
x = Dense(64, activation='relu')(x)
```

- Define output layer

```sh

```
