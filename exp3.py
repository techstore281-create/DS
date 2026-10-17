
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# Step 1: Define configuration
IMG_SIZE = (64, 64)
BATCH_SIZE = 32
SEED = 42

DATASET_DIR = "COVID-19_Radiography_Dataset"

COVID_DIR = os.path.join(
    DATASET_DIR, "COVID", "images"
)

NORMAL_DIR = os.path.join(
    DATASET_DIR, "Normal", "images"
)


# Step 2: Get image paths from folders
def get_images(folder):
    extensions = (".png", ".jpg", ".jpeg")

    return [
        os.path.join(folder, file)
        for file in os.listdir(folder)
        if file.lower().endswith(extensions)
    ]


covid_images = get_images(COVID_DIR)
normal_images = get_images(NORMAL_DIR)

print("COVID images:", len(covid_images))
print("Normal images:", len(normal_images))


# Step 3: Create labels
image_paths = normal_images + covid_images

labels = (
    [0] * len(normal_images)
    + [1] * len(covid_images)
)


# Step 4: Split data into training and testing sets
train_paths, test_paths, train_labels, test_labels = (
    train_test_split(
        image_paths,
        labels,
        test_size=0.20,
        random_state=SEED,
        stratify=labels
    )
)


# Step 5: Load and preprocess images
def load_image(path, label):
    image = tf.io.read_file(path)

    image = tf.image.decode_png(
        image,
        channels=3
    )

    image = tf.image.resize(
        image,
        IMG_SIZE
    )

    return image, label


# Create TensorFlow datasets
train_ds = tf.data.Dataset.from_tensor_slices(
    (train_paths, train_labels)
)

test_ds = tf.data.Dataset.from_tensor_slices(
    (test_paths, test_labels)
)

train_ds = train_ds.map(
    load_image,
    num_parallel_calls=tf.data.AUTOTUNE
)

test_ds = test_ds.map(
    load_image,
    num_parallel_calls=tf.data.AUTOTUNE
)


# Step 6: Shuffle, batch, and optimize datasets
train_ds = train_ds.shuffle(
    buffer_size=len(train_paths),
    seed=SEED
)

train_ds = train_ds.batch(BATCH_SIZE)
test_ds = test_ds.batch(BATCH_SIZE)

train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
test_ds = test_ds.prefetch(tf.data.AUTOTUNE)


# Step 7: Build the CNN model
model = models.Sequential([
    layers.Input(shape=(64, 64, 3)),

    layers.Rescaling(1.0 / 255),

    layers.Conv2D(
        16, (3, 3), activation="relu"
    ),

    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(
        32, (3, 3), activation="relu"
    ),

    layers.MaxPooling2D((2, 2)),

    layers.Flatten(),

    layers.Dense(
        32, activation="relu"
    ),

    layers.Dropout(0.2),

    layers.Dense(
        1, activation="sigmoid"
    )
])


# Step 8: Compile the model
model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.summary()


# Step 9: Calculate class weights
normal_count = len(normal_images)
covid_count = len(covid_images)

total = normal_count + covid_count

class_weights = {
    0: total / (2 * normal_count),
    1: total / (2 * covid_count)
}

print("\nClass Weights:")
print(class_weights)


# Step 10: Train the model
history = model.fit(
    train_ds,
    validation_data=test_ds,
    epochs=4,
    class_weight=class_weights
)


# Step 11: Evaluate predictions
y_true = np.concatenate([
    y.numpy().flatten()
    for x, y in test_ds
], axis=0).astype(int)

y_pred_probs = model.predict(test_ds)

y_pred = (
    y_pred_probs.flatten() > 0.5
).astype(int)


# Step 12: Display classification report
print("\n--- Balanced Classification Report ---")

print(
    classification_report(
        y_true,
        y_pred,
        labels=[0, 1],
        target_names=["Normal", "COVID-19"],
        zero_division=0
    )
)


# Step 13: Plot confusion matrix
cm = confusion_matrix(
    y_true,
    y_pred,
    labels=[0, 1]
)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Normal", "COVID-19"]
)

fig, ax = plt.subplots(figsize=(4, 3))

disp.plot(
    cmap="Blues",
    values_format="d",
    ax=ax
)

plt.title("COVID-19 X-Ray Diagnosis Matrix")
plt.tight_layout()
plt.show()


# Step 14: Plot training and validation accuracy
plt.figure(figsize=(6, 4))

plt.plot(
    history.history["accuracy"],
    label="Training Accuracy"
)

plt.plot(
    history.history["val_accuracy"],
    label="Validation Accuracy"
)

plt.title("CNN Model Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.tight_layout()
plt.show()
