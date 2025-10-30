import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# ===============================
# Load Dataset
# ===============================
file_path = "food_dataset.xlsx"
df_original = pd.read_excel(file_path, sheet_name="FOOD-DATA-GROUP1")

# Keep a copy for visualization comparison
df_before = df_original.copy()

# ===============================
# 1. Data Cleaning
# ===============================
if 'Unnamed: 0' in df_original.columns:
    df_original = df_original.drop(columns=['Unnamed: 0'])

if 'food' in df_original.columns:
    df_original['food'] = df_original['food'].astype(str).str.strip().str.lower()

num_cols = [c for c in df_original.columns if c != 'food']
for c in num_cols:
    df_original[c] = pd.to_numeric(df_original[c], errors='coerce')

df_original = df_original.drop_duplicates()

numeric_cols = df_original.select_dtypes(include=[np.number]).columns.tolist()
imputer = SimpleImputer(strategy='median')
df_original[numeric_cols] = imputer.fit_transform(df_original[numeric_cols])

df_cleaned = df_original.copy()

# ===============================
# 2. Data Integration
# ===============================
# Suppose there is another sheet or dataset with related info
# For example, integrating FOOD-DATA-GROUP2 (optional step)
try:
    df_extra = pd.read_excel(file_path, sheet_name="FOOD-DATA-GROUP2")
    df_integrated = pd.concat([df_cleaned, df_extra], ignore_index=True)
except:
    df_integrated = df_cleaned.copy()

# ===============================
# 3. Data Reduction
# ===============================
columns_to_keep = ['Caloric Value', 'Fat', 'Protein', 'Carbohydrates', 'Nutrition Density']
df_reduced = df_integrated[columns_to_keep]

# ===============================
# 4. Data Transformation (Scaling)
# ===============================
scaler = MinMaxScaler()
df_scaled = df_reduced.copy()
df_scaled[columns_to_keep] = scaler.fit_transform(df_scaled[columns_to_keep])

# ===============================
# 5. Data Discretization
# ===============================
# Convert continuous numerical values into categories (bins)
# Example: Discretize "Caloric Value" into low, medium, high
df_discretized = df_scaled.copy()
df_discretized['Calorie_Level'] = pd.cut(
    df_discretized['Caloric Value'],
    bins=3,
    labels=['Low', 'Medium', 'High']
)

# ===============================
# 6. Data Augmentation
# ===============================
# Add small variations to increase dataset size
df_augmented = df_discretized.copy()
df_augmented['Fat'] = df_augmented['Fat'] * np.random.uniform(1.05, 1.15, len(df_augmented))
df_augmented['Protein'] = df_augmented['Protein'] * np.random.uniform(0.9, 1.1, len(df_augmented))

# Combine original + augmented data
df_final = pd.concat([df_discretized, df_augmented], ignore_index=True)

# ===============================
# Visualization Section - Histograms
# ===============================
plt.figure(figsize=(15, 12))

for i, col in enumerate(columns_to_keep):
    plt.subplot(3, 2, i + 1)
    plt.hist(df_reduced[col], bins=15, alpha=0.5, label='Before Scaling/Cleaning')
    plt.hist(df_scaled[col], bins=15, alpha=0.5, label='Scaled')
    plt.hist(df_final[col], bins=15, alpha=0.5, label='After Augmentation')
    plt.title(f"Distribution of {col}")
    plt.xlabel(col)
    plt.ylabel("Frequency")
    plt.legend()

plt.tight_layout()
plt.show()
