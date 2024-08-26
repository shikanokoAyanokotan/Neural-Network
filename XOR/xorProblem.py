import numpy as np
import tensorflow as tf

# Data for XOR problem
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
Y = np.array([[0], [1], [1], [0]], dtype=np.float32)

# Define the neural network model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(2, input_dim=2, activation='relu'), # Hidden layer with 2 neurons
    tf.keras.layers.Dense(1, activation='sigmoid') # Output layer with 1 neuron
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X, Y, epochs=2000, verbose=0)

# Evaluate the model
loss, accuracy = model.evaluate(X, Y)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Test the model
predictions = model.predict(X)
print("Predictions:")
print(np.round(predictions))
