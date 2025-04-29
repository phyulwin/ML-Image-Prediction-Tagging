# MODEL TRAINING PHASE 1

# # COCO-2017 80-Category Image Classifier
# 
# A from-scratch CNN for multi-label classification on COCO-2017.

# Install any missing packages
# !pip install fiftyone tensorflow numpy matplotlib pillow opencv-python

import random
import fiftyone.zoo as foz
import numpy as np
import tensorflow as tf
import os
import matplotlib.pyplot as plt
from tensorflow.keras import Sequential, Input
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.callbacks import TensorBoard

# ## Configuration

# How many COCO samples to load (None = all ~82k)
MAX_SAMPLES = 18000

# Training hyperparams
BATCH_SIZE = 16
EPOCHS     = 20
IMG_SIZE   = (256, 256)

# For reproducibility
random.seed(42)
tf.random.set_seed(42)

# ## Load & split COCO-2017

# 1) Load COCO-2017 training split via FiftyOne
dataset = foz.load_zoo_dataset(
    "coco-2017",
    split="train",
    max_samples=MAX_SAMPLES
)

# 2) Shuffle & split into train / val / test
samples = list(dataset)
random.shuffle(samples)
n = len(samples)
n_train = int(0.7 * n)
n_val   = int(0.2 * n)

train_samples = samples[:n_train]
val_samples   = samples[n_train:n_train + n_val]
test_samples  = samples[n_train + n_val:]

# ## Build class map & tf.data pipelines

# COCO has 80 classes
class_list   = dataset.default_classes
num_classes  = len(class_list)
class_to_idx = {c: i for i, c in enumerate(class_list)}

def sample_to_path_label(sample):
    path = sample.filepath
    lbl  = np.zeros(num_classes, dtype=np.int32)
    
    dets = getattr(sample.ground_truth, "detections", None)
    if dets:
        for det in dets:
            lbl[class_to_idx[det.label]] = 1
    return path, lbl

# Gather paths & labels
train_paths, train_labels = zip(*(sample_to_path_label(s) for s in train_samples))
val_paths,   val_labels   = zip(*(sample_to_path_label(s) for s in val_samples))
test_paths,  test_labels  = zip(*(sample_to_path_label(s) for s in test_samples))

def make_ds(paths, labels):
    ds = tf.data.Dataset.from_tensor_slices((list(paths), list(labels)))
    def _load(path, label):
        img = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, IMG_SIZE)
        img = img / 255.0
        return img, label
    return ds.map(_load, num_parallel_calls=tf.data.AUTOTUNE) \
             .batch(BATCH_SIZE) \
             .prefetch(tf.data.AUTOTUNE)

train_ds = make_ds(train_paths, train_labels)
val_ds   = make_ds(val_paths,   val_labels)
test_ds  = make_ds(test_paths,  test_labels)

model = Sequential([
    Input(shape=(*IMG_SIZE, 3)),
    # Conv block 1
    Conv2D(16, (3,3), activation='relu'),
    MaxPooling2D(),
    # Conv block 2
    Conv2D(32, (3,3), activation='relu'),
    MaxPooling2D(),
    # Conv block 3
    Conv2D(16, (3,3), activation='relu'),
    MaxPooling2D(),
    # Classifier head
    Flatten(),
    Dense(256, activation='relu'),
    Dense(num_classes, activation='sigmoid')   # 80-way multi-label
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',     # multi-label
    metrics=['binary_accuracy']
)

model.summary()

logdir = 'logs'
tensorboard_cb = TensorBoard(log_dir=logdir)

history = model.fit(
    train_ds,
    epochs=EPOCHS,
    validation_data=val_ds,
    callbacks=[tensorboard_cb]
)

plt.figure()
plt.plot(history.history['loss'], label='train loss')
plt.plot(history.history['val_loss'], label='val loss')
plt.legend()
plt.title('Loss')

plt.figure()
plt.plot(history.history['binary_accuracy'], label='train acc')
plt.plot(history.history['val_binary_accuracy'], label='val acc')
plt.legend()
plt.title('Accuracy')


test_loss, test_acc = model.evaluate(test_ds)
print(f"Test loss: {test_loss:.4f}, Test accuracy: {test_acc:.4f}")

os.makedirs('models', exist_ok=True)
model.save('models/coco80_cnn.h5')
print("Model saved to models/coco80_cnn.h5") # renamed to modelPhase1.h5

# Test loss: 0.6515, Test accuracy: 0.9559
# WARNING:absl:You are saving your model as an HDF5 file via `model.save()` or `keras.saving.save_model(model)`. This file format is considered legacy. We recommend using instead the native Keras format, e.g. `model.save('my_model.keras')` or `keras.saving.save_model(model, 'my_model.keras')`. 
# Model saved to models/coco80_cnn.h5