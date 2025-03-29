import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import boston_housing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Load the Boston Housing Dataset
(X_train, y_train), (X_test, y_test) = boston_housing.load_data()

# Standardize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Define a custom cascade correlation layer
class CascadeCorrelationLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(CascadeCorrelationLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.dense = tf.keras.layers.Dense(1, activation='linear')
        self.built = True

    def call(self, inputs):
        return self.dense(inputs)

# Create a Cascade Correlation model
model = tf.keras.Sequential()

# Input layer
model.add(tf.keras.layers.Input(shape=(X_train.shape[1],)))

# Initial hidden layer
model.add(CascadeCorrelationLayer())

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model (you can adjust the number of epochs and batch size)
model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2)

# Evaluate the model on the test set
loss = model.evaluate(X_test, y_test)
print(f"Test loss: {loss:.4f}")
