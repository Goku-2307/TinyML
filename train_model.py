import pandas as pd
import numpy as np
import json
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Make results reproducible
np.random.seed(42)
tf.random.set_seed(42)

# 1. Load dataset
df = pd.read_csv("filtravyn_water_quality_dataset.csv")

# 2. Select model inputs in a fixed order
FEATURES = [
    "tds_ppm",
    "turbidity_NTU_simulated",
    "temperature_C",
    "flow_rate_L_min_simulated"
]

TARGET = "water_quality_label_synthetic"

X = df[FEATURES].values.astype(np.float32)

# Fixed class order for later ESP32 implementation
class_names = ["Good", "Moderate", "Poor"]

class_to_number = {
    "Good": 0,
    "Moderate": 1,
    "Poor": 2
}

y = df[TARGET].map(class_to_number).values.astype(np.int32)

if np.isnan(y).any():
    raise ValueError("Unknown target label found in dataset.")

# 3. Split dataset into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# 4. Normalize the input features
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train).astype(np.float32)
X_test_scaled = scaler.transform(X_test).astype(np.float32)

# 5. Build neural network
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(4,)),
    tf.keras.layers.Dense(16, activation="relu"),
    tf.keras.layers.Dense(8, activation="relu"),
    tf.keras.layers.Dense(3, activation="softmax")
])

# 6. Configure training
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# 7. Train model
print("\nSTARTING MODEL TRAINING...\n")

history = model.fit(
    X_train_scaled,
    y_train,
    epochs=100,
    batch_size=16,
    validation_split=0.20,
    verbose=1
)

# 8. Evaluate on unseen test data
loss, keras_accuracy = model.evaluate(
    X_test_scaled,
    y_test,
    verbose=0
)

predictions = np.argmax(
    model.predict(X_test_scaled, verbose=0),
    axis=1
)

accuracy = accuracy_score(y_test, predictions)

print("\n========== MODEL RESULTS ==========")
print("Test loss:", round(float(loss), 4))
print("Test accuracy:", round(float(accuracy * 100), 2), "%")

print("\nCLASSIFICATION REPORT:")
print(classification_report(
    y_test,
    predictions,
    labels=[0, 1, 2],
    target_names=class_names,
    zero_division=0
))

print("CONFUSION MATRIX:")
print(confusion_matrix(y_test, predictions, labels=[0, 1, 2]))

# 9. Save trained model
model.save("filtravyn_model.keras")

# 10. Save scaler parameters and class order
scaler_data = {
    "features": FEATURES,
    "mean": scaler.mean_.tolist(),
    "scale": scaler.scale_.tolist(),
    "classes": class_names
}

with open("filtravyn_scaler.json", "w") as file:
    json.dump(scaler_data, file, indent=4)

print("\nFILES SAVED:")
print("- filtravyn_model.keras")
print("- filtravyn_scaler.json")

print("\nSTAGE 2 TRAINING COMPLETED.")