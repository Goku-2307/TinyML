import numpy as np
import tensorflow as tf
import json

print("FILTRAVYN PYTHON REFERENCE TEST")
print("--------------------------------")

# Load trained model
model = tf.keras.models.load_model("filtravyn_model.keras")
print("Model loaded successfully")

# Load scaler
with open("filtravyn_scaler.json", "r") as f:
    scaler = json.load(f)

mean = np.array(scaler["mean"], dtype=np.float32)
scale = np.array(scaler["scale"], dtype=np.float32)

# Input features in training order
sample = np.array([[
    793.60,
    11.65,
    23.46,
    1.72
]], dtype=np.float32)

# Normalize input
normalized = (sample - mean) / scale

# Predict
probabilities = model.predict(normalized, verbose=0)[0]

classes = ["Good", "Moderate", "Poor"]

print("\nINPUT FEATURES:")
print(sample[0])

print("\nNORMALIZED FEATURES:")
print(normalized[0])

print("\nPREDICTION PROBABILITIES:")

for name, probability in zip(classes, probabilities):
    print(f"{name}: {probability:.8f}")

print("\nPREDICTED CLASS:")
print(classes[np.argmax(probabilities)])

print("\nREFERENCE TEST COMPLETED")