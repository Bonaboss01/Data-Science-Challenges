# Can be used for churn, fraud or stock prediction
"""
binary_classification_tf.py

A minimal TensorFlow example:
- Generate synthetic data
- Build a neural network with Keras
- Train with EarlyStopping
- Evaluate and make predictions
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split

# =========================
# 1. REPRODUCIBILITY
# =========================
np.random.seed(42)
tf.random.set_seed(42)

# =========================
# 2. GENERATE SYNTHETIC DATA
# =========================
# We'll create a non-linear binary classification problem.

def generate_data(n_samples=2000):
    """
    Generate 2D points (x1, x2) with a non-linear decision boundary:
    label = 1 if x2 > sin(x1 * 3) + noise, else 0
    """
    x1 = np.random.uniform(-2, 2, size=n_samples)
    x2 = np.random.uniform(-2, 2, size=n_samples)

    # Curvy boundary using a sine wave
    boundary = np.sin(3 * x1)
    noise = np.random.normal(0, 0.3, size=n_samples)

    y = (x2 > boundary + noise).astype(int)

    X = np.column_stack([x1, x2])
    return X, y

X, y = generate_data(n_samples=3000)

# Train/validation/test split: 60% / 20% / 20%
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp
)
# (0.25 of 0.8 = 0.2, so overall 60/20/20)

print("Train shape:", X_train.shape, y_train.shape)
print("Val shape:  ", X_val.shape, y_val.shape)
print("Test shape: ", X_test.shape, y_test.shape)

# =========================
# 3. BUILD THE MODEL
# =========================
# Simple fully-connected network for binary classification.

def build_model(input_dim: int = 2) -> keras.Model:
    model = keras.Sequential(
        [
            layers.Input(shape=(input_dim,)),
            layers.Dense(32, activation="relu"),
            layers.Dense(32, activation="relu"),
            layers.Dense(1, activation="sigmoid"),  # Binary output (0–1 probability)
        ]
    )

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model

model = build_model(input_dim=X_train.shape[1])
model.summary()

# =========================
# 4. TRAIN THE MODEL
# =========================
# We use EarlyStopping to prevent overfitting.

early_stopping = keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=10,            # epochs with no improvement before stopping
    restore_best_weights=True,
    verbose=1,
)

history = model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=200,
    batch_size=32,
    callbacks=[early_stopping],
    verbose=1,
)

# =========================
# 5. EVALUATE ON TEST SET
# =========================
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"\nTest Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_acc:.4f}")

# =========================
# 6. MAKE SOME PREDICTIONS
# =========================
# Take a few samples from the test set and show predicted probabilities.

sample_X = X_test[:5]
sample_y_true = y_test[:5]

y_probs = model.predict(sample_X)          # probabilities
y_pred = (y_probs > 0.5).astype(int)      # class labels 0/1

print("\nSample predictions:")
for i in range(len(sample_X)):
    x1, x2 = sample_X[i]
    print(
        f"Point(x1={x1:.2f}, x2={x2:.2f}) | "
        f"True={sample_y_true[i]} | "
        f"Pred={int(y_pred[i])} | "
        f"Prob={y_probs[i][0]:.3f}"
    )

# =========================
# 7. OPTIONAL: SAVE MODEL
# =========================
# You can save the trained model for later use.
model.save("binary_classifier_tf.keras")
print("\nModel saved to 'binary_classifier_tf.keras'")
