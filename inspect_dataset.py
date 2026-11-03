
import pandas as pd
import numpy as np
import os

FILE = "filtravyn_water_quality_dataset.csv"

print("=" * 55)
print("FILTRAVYN - DATASET INSPECTION")
print("=" * 55)

if not os.path.exists(FILE):
    print("ERROR: Dataset file not found:", FILE)
    raise SystemExit

df = pd.read_csv(FILE)

print("\n1. DATASET DIMENSIONS")
print("Rows:", df.shape[0])
print("Columns:", df.shape[1])

print("\n2. COLUMN NAMES")
for col in df.columns:
    print("-", col)

print("\n3. FIRST FIVE ROWS")
print(df.head().to_string(index=False))

print("\n4. DATA TYPES")
print(df.dtypes)

print("\n5. MISSING VALUES")
print(df.isnull().sum())

print("\n6. DUPLICATE ROWS")
print(df.duplicated().sum())

print("\n7. WATER QUALITY LABEL COUNTS")
label = "water_quality_label_synthetic"

if label in df.columns:
    print(df[label].value_counts())
else:
    print("ERROR: Target label column not found.")

print("\n8. NUMERIC DATA SUMMARY")
print(df.describe().to_string())

print("\n9. INFINITE VALUES")
numeric = df.select_dtypes(include=[np.number])
print(np.isinf(numeric).sum())

print("\nINSPECTION COMPLETED")
