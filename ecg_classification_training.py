import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

# 1. Data Loading and Preprocessing
normal_df = pd.read_csv('/dataset/ptbdb_normal.csv', header=None)
abnormal_df = pd.read_csv('/dataset/ptbdb_abnormal.csv', header=None)

# Assign labels: 0 = normal, 1 = abnormal
normal_df.iloc[:, -1] = 0
abnormal_df.iloc[:, -1] = 1

# Combine data
df = pd.concat([normal_df, abnormal_df], axis=0).sample(frac=1).reset_index(drop=True)

# Split data
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

# Normalize data
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split into train/test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. Model Training (DNN)
model = keras.Sequential([
    keras.Input(shape=(X_train.shape[1],)),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(32, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

history = model.fit(X_train, y_train, epochs=30, batch_size=32, validation_data=(X_test, y_test))

model.save('my_model.keras')

joblib.dump(scaler, 'scaler.save')
scaler = joblib.load('scaler.save')

# Plot Accuracy and Validation Accuracy
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.show()

# Test Acuracy
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_acc:.4f}")