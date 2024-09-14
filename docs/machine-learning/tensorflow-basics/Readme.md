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

Dropout is a regularization technique that prevents overfitting in neural networks. Overfitting occurs when a model learns the training data too accurately, including noise and outliers, resulting in poor generalization to new data. Technically, dropout during the training of a neural network involves randomly selecting a subset of neurons (along with their connections) for each training iteration and temporarily removing them from the network. The probability with which a neuron is switched off is determined by the dropout rate parameter, which is usually between 0.2 and 0.5. This forces the network to learn redundant representations of the data as it cannot rely on the presence of specific neurons.

- Activation function:

In itself, there are, for example, tangent hyperbolic (tanh), sigmoid or relu activation functions. For large networks, the relu function is often used in the fc layers because it also weights values close to 0 very well and is also very fast. 

- Layers:

Choose a sensible structure. Too many layers often lead to a model being overfitted. You should therefore test several architectures together.

- Input layer:

This must be adapted to flatten the data depending on the input. 


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
outputs = Dense(1)(x)  # Für Regression
```

- create model

```sh
model = Model(inputs=inputs, outputs=outputs)
```

