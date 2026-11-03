import tensorflow as tf
import numpy as np
import json

# Load trained model
model = tf.keras.models.load_model("filtravyn_model.keras")

# Load scaler
with open("filtravyn_scaler.json", "r") as f:
    scaler = json.load(f)

# Extract Dense layer weights and biases
dense_layers = [
    layer for layer in model.layers
    if isinstance(layer, tf.keras.layers.Dense)
]

if len(dense_layers) != 3:
    raise ValueError("Expected exactly 3 Dense layers.")

expected_shapes = [
    ((4, 16), (16,)),
    ((16, 8), (8,)),
    ((8, 3), (3,))
]

for i, (layer, expected) in enumerate(zip(dense_layers, expected_shapes)):
    weights, biases = layer.get_weights()

    if weights.shape != expected[0] or biases.shape != expected[1]:
        raise ValueError(
            f"Unexpected shape in layer {i + 1}: "
            f"{weights.shape}, {biases.shape}"
        )

def format_array(name, values):
    values = np.asarray(values, dtype=np.float32).flatten()
    contents = ", ".join(f"{v:.9g}f" for v in values)
    return (
        f"const float {name}[{len(values)}] = "
        f"{{{contents}}};\n"
    )

# Build C++ header
header = """#ifndef FILTRAVYN_MODEL_H
#define FILTRAVYN_MODEL_H

// Automatically exported from the trained Keras model.
// Dense weights are stored row-major: input index first,
// output index second.

"""

names = [
    ("W1", "b1"),
    ("W2", "b2"),
    ("W3", "b3")
]

for layer, (weight_name, bias_name) in zip(dense_layers, names):
    weights, biases = layer.get_weights()

    header += format_array(weight_name, weights)
    header += format_array(bias_name, biases)
    header += "\n"

header += format_array("FEATURE_MEAN", scaler["mean"])
header += format_array("FEATURE_SCALE", scaler["scale"])

header += """
const char* CLASS_NAMES[3] = {
    "Good",
    "Moderate",
    "Poor"
};

#endif
"""

with open("filtravyn_model.h", "w") as f:
    f.write(header)

print("MODEL EXPORT SUCCESSFUL")
print("Dense layers:", len(dense_layers))

for i, layer in enumerate(dense_layers, start=1):
    w, b = layer.get_weights()
    print(f"Layer {i}: weights={w.shape}, biases={b.shape}")

print("\nScaler mean:", scaler["mean"])
print("Scaler scale:", scaler["scale"])

print("\nGenerated file: filtravyn_model.h")