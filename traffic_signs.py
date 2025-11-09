import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
from PIL import Image

# Paths
TRAIN_DIR = "train"
TEST_DIR = "test"
SIGNNAMES = "meta/signnames.csv"

# Data loading (basic example)
def load_data(data_dir):
    images = []
    labels = []
    for label in os.listdir(data_dir):
        img_dir = os.path.join(data_dir, label)
        for img_name in os.listdir(img_dir):
            img_path = os.path.join(img_dir, img_name)
            img = Image.open(img_path).resize((30, 30))
            images.append(np.array(img))
            labels.append(int(label))
    return np.array(images), np.array(labels)

print("Loading training data...")
X_train, y_train = load_data(TRAIN_DIR)

print("Loading test data...")
X_test, y_test = load_data(TEST_DIR)

# Model definition
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(30, 30, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(43, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

print("Training model...")
model.fit(X_train, y_train, epochs=10, validation_split=0.2)

print("Evaluating model...")
test_loss, test_acc = model.evaluate(X_test, y_test)
print("Test accuracy:", test_acc)

# Save the model
model.save('model/traffic_sign_model.h5')
