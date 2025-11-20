import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt

# ===============================
# 1. Load Dataset
# ===============================
file_path = "food_dataset.xlsx"
df_original = pd.read_excel(file_path, sheet_name="FOOD-DATA-GROUP1")

# ===============================
# 2. Data Cleaning
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
# 3. Data Integration (Optional)
# ===============================
try:
    df_extra = pd.read_excel(file_path, sheet_name="FOOD-DATA-GROUP2")
    df_integrated = pd.concat([df_cleaned, df_extra], ignore_index=True)
except:
    df_integrated = df_cleaned.copy()

# ===============================
# 4. Data Reduction
# ===============================
columns_to_keep = ['Caloric Value', 'Fat', 'Protein', 'Carbohydrates', 'Nutrition Density']
df_reduced = df_integrated[columns_to_keep]

# ===============================
# 5. Data Transformation (Scaling)
# ===============================
scaler = MinMaxScaler()
df_scaled = df_reduced.copy()
df_scaled[columns_to_keep] = scaler.fit_transform(df_scaled[columns_to_keep])

# ===============================
# 6. Data Discretization (Balanced bins)
# ===============================
# Use qcut to create bins with roughly equal number of samples
df_discretized = df_scaled.copy()
df_discretized['Calorie_Level'] = pd.qcut(
    df_discretized['Caloric Value'],
    q=3,
    labels=['Low', 'Medium', 'High']
)

# ===============================
# 7. Split Data Before Augmentation
# ===============================
X = df_discretized[columns_to_keep]
y = df_discretized['Calorie_Level']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ===============================
# 8. Data Augmentation on Training Set Only
# ===============================
X_train_aug = X_train.copy()
y_train_aug = y_train.copy()

# Small random variations
X_train_aug['Fat'] = X_train_aug['Fat'] * np.random.uniform(1.05, 1.15, len(X_train_aug))
X_train_aug['Protein'] = X_train_aug['Protein'] * np.random.uniform(0.9, 1.1, len(X_train_aug))

# Combine original + augmented training data
X_train_final = pd.concat([X_train, X_train_aug], ignore_index=True)
y_train_final = pd.concat([y_train, y_train_aug], ignore_index=True)

# ===============================
# 9. Model Training
# ===============================
model = RandomForestClassifier(n_estimators=80, max_depth=5, random_state=42)
model.fit(X_train_final, y_train_final)

# ===============================
# 10. Predictions & Evaluation
# ===============================
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Test Set Accuracy: {accuracy:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# ===============================
# 11. Cross-Validation Accuracy
# ===============================
cv_scores = cross_val_score(model, X_train_final, y_train_final, cv=5)
print(f"\n5-Fold Cross-Validation Accuracy: {cv_scores.mean():.4f}")

# ===============================
# 12. Visualization
# ===============================
plt.figure(figsize=(15, 12))
for i, col in enumerate(columns_to_keep):
    plt.subplot(3, 2, i + 1)
    plt.hist(df_reduced[col], bins=15, alpha=0.5, label='Before Scaling/Cleaning')
    plt.hist(df_scaled[col], bins=15, alpha=0.5, label='Scaled')
    plt.hist(X_train_final[col], bins=15, alpha=0.5, label='After Augmentation')
    plt.title(f"Distribution of {col}")
    plt.xlabel(col)
    plt.ylabel("Frequency")
    plt.legend()
plt.tight_layout()
plt.show()
