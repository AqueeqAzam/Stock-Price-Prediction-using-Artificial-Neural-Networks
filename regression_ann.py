import tensorflow as tf
import numpy as np

# 🧱 Input data: House area (in sq ft) and their respective prices (in thousands)
x = tf.constant([[1000.0], [1500.0], [2500.0]], dtype=tf.float32)  # Area
y = tf.constant([[100.0], [150.0], [200.0]], dtype=tf.float32)     # Price

# 🧠 Model: A simple neural network with 1 layer & 1 neuron
# Behind the scenes:
# - 1 input → 1 output (Price = Weight * Area + Bias)
# - TensorFlow automatically initializes the weight and bias randomly.
model = tf.keras.Sequential([
    tf.keras.layers.Dense(units=3, input_shape=[1])  # Linear regression layer
    # This means: 1 neuron, expecting 1 input feature (area)
])

# ⚙️ Compile the model with a loss function and an optimizer
model.compile(
    optimizer='sgd',                     # SGD = Stochastic Gradient Descent
    # This adjusts the weight and bias using gradients to minimize error.
    loss='mean_squared_error'           # Measures how far off the predictions are (penalizes large errors more)
    # The model learns by reducing this error over time.
)

# 🚀 Train the model with the data
print('\n🏋️‍♂️ Training model...')
model.fit(x, y, epochs=100, verbose=0)  # Do 100 rounds of training (weight & bias updates)

# Behind the scenes:
# For each epoch:
# - Predict price from input area using current weights/bias
# - Compare prediction with actual price (loss)
# - Calculate gradients (how much to change weight & bias)
# - Update weight & bias to reduce the error
# - Repeat

# 🔮 Make a prediction: What should be the price of a 3000 sq. ft. house?
area = tf.constant([[3000.0]])  # New input
prediction = model.predict(area)  # Predict using the trained model (uses learned weight & bias)

# 💬 Show the result
print(f"\n🏡 Predicted price for 3000 sq ft house: ${prediction[0][0]:.2f}k")

