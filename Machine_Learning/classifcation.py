# ===========================================
# FULL CLASSIFICATION MODELLING WORKFLOW
# Demonstrates:
# - Data preparation & EDA
# - Multiple models (LogReg, RF, KNN, SVM)
# - Model comparison using accuracy
# - Selecting & "deploying" best model
# ===========================================

# -------- 1. IMPORT LIBRARIES --------
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

import joblib  # for saving the best model


# -------- 2. CREATE / LOAD DATA --------
# Here we create a synthetic binary classification dataset.
# In real life, you would replace this with: df = pd.read_csv("your_data.csv")
X, y = make_classification(
    n_samples=1000,
    n_features=10,
    n_informative=6,
    n_redundant=2,
    n_clusters_per_class=2,
    random_state=42
)

# Put into a DataFrame for convenience
feature_names = [f"feature_{i}" for i in range(X.shape[1])]
df = pd.DataFrame(X, columns=feature_names)
df["target"] = y

print("Data preview:")
print(df.head())
print("\nClass balance:")
print(df["target"].value_counts())


# -------- 3. BASIC EDA --------
print("\nSummary statistics:")
print(df.describe())

# Plot class distribution
plt.figure(figsize=(4,3))
sns.countplot(x="target", data=df)
plt.title("Target Class Distribution")
plt.show()

# Optional: correlation heatmap
plt.figure(figsize=(8,6))
sns.heatmap(df.corr(), cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()


# -------- 4. TRAIN-TEST SPLIT --------
X = df[feature_names]
y = df["target"]

# Split into train (70%), validation (15%), test (15%)
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.15, random_state=42, stratify=y
)

X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.1765,  # 0.1765 * 0.85 ≈ 0.15 of original
    random_state=42,
    stratify=y_temp
)

print(f"\nTrain size: {X_train.shape[0]}")
print(f"Validation size: {X_val.shape[0]}")
print(f"Test size: {X_test.shape[0]}")


# -------- 5. FEATURE SCALING --------
# Many models (KNN, SVM, Logistic Regression) benefit from scaling.
scaler = StandardScaler()

# Fit only on training data
X_train_scaled = scaler.fit_transform(X_train)

# Transform validation and test using the same scaler
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)


# -------- 6. DEFINE CANDIDATE MODELS --------
# We will try multiple models and select the best based on validation accuracy.

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Random Forest": RandomForestClassifier(
        n_estimators=200, random_state=42
    ),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "SVM (RBF)": SVC(kernel="rbf", probability=True, random_state=42)
}

val_scores = {}  # to store validation accuracy for each model


# -------- 7. TRAIN & EVALUATE EACH MODEL ON VALIDATION SET --------
for name, model in models.items():
    # Use scaled features
    model.fit(X_train_scaled, y_train)
    y_val_pred = model.predict(X_val_scaled)
    acc = accuracy_score(y_val, y_val_pred)
    val_scores[name] = acc
    print(f"{name} - Validation Accuracy: {acc:.4f}")

# Identify best model based on validation accuracy
best_model_name = max(val_scores, key=val_scores.get)
best_model = models[best_model_name]

print("\nBest model based on validation accuracy:", best_model_name)
print("Best validation accuracy:", val_scores[best_model_name])


# -------- 8. RE-TRAIN BEST MODEL ON FULL TRAINING DATA (TRAIN + VAL) --------
# In practice, after choosing the best model,
# we retrain it using all available non-test data to maximize learning.

# Combine train + validation
X_full_train = np.vstack((X_train_scaled, X_val_scaled))
y_full_train = np.concatenate((y_train, y_val))

# Refit the best model
best_model.fit(X_full_train, y_full_train)


# -------- 9. FINAL EVALUATION ON TEST SET --------
y_test_pred = best_model.predict(X_test_scaled)
test_accuracy = accuracy_score(y_test, y_test_pred)

print("\n=== FINAL TEST PERFORMANCE OF BEST MODEL ===")
print(f"Best Model: {best_model_name}")
print(f"Test Accuracy: {test_accuracy:.4f}\n")

# Confusion Matrix
cm = confusion_matrix(y_test, y_test_pred)
plt.figure(figsize=(4,3))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title(f"Confusion Matrix - {best_model_name}")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Classification Report (precision, recall, f1-score)
print("Classification Report:")
print(classification_report(y_test, y_test_pred))


# -------- 10. "DEPLOY" BEST MODEL --------
# In a real-world scenario, deployment could mean:
#  - Saving the trained model to disk (e.g., .pkl)
#  - Loading it in a web service / API
#  - Integrating into a batch or real-time pipeline

# Here we simulate deployment by saving:
model_filename = "best_classification_model.pkl"
scaler_filename = "scaler.pkl"  # Save scaler as well so we can transform new data

joblib.dump(best_model, model_filename)
joblib.dump(scaler, scaler_filename)

print(f"\nBest model saved to: {model_filename}")
print(f"Scaler saved to: {scaler_filename}")


# -------- 11. EXAMPLE: USING THE DEPLOYED MODEL ON NEW DATA --------
# Simulate a few new observations
new_samples = np.array([
    [0.5, -1.2, 0.3, 1.1, -0.7, 0.9, 1.5, -0.4, 0.2, -1.0],
    [-0.8, 0.6, -1.1, 0.3, 1.2, -0.5, -0.3, 0.8, -0.2, 0.4]
])

# Load model & scaler (this is what would happen in production)
loaded_scaler = joblib.load(scaler_filename)
loaded_model = joblib.load(model_filename)

# Scale new data
new_samples_scaled = loaded_scaler.transform(new_samples)

# Predict class
new_preds = loaded_model.predict(new_samples_scaled)
new_probas = loaded_model.predict_proba(new_samples_scaled)

print("\nPredictions for new samples:")
for i, (cls, prob) in enumerate(zip(new_preds, new_probas)):
    print(f"Sample {i+1}: Predicted class = {cls}, Probabilities = {prob}")

