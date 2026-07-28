#!/usr/bin/env python3
"""
Smart Harvest AI — Plant Disease Detection Model
Transfer learning with MobileNetV3Small on PlantVillage dataset.
Optimised for MacBook M1 8GB. Exports TFLite for Render deployment.
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from pathlib import Path
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# ─── Config ───────────────────────────────────────────────────────────────────
DATASET_DIR      = "./PlantVillage/PlantVillage"
MODELS_DIR       = Path("./models")
OUTPUTS_DIR      = Path("./outputs")
CACHE_DIR        = Path("./cache")
CHECKPOINT_PATH  = str(MODELS_DIR / "best_model.keras")
TFLITE_PATH      = str(MODELS_DIR / "disease_model.tflite")
LABELS_PATH      = str(MODELS_DIR / "class_labels.json")

IMAGE_SIZE       = (192, 192)
BATCH_SIZE       = 16
ALPHA            = 1.0           # full MobileNetV3Small capacity
PHASE1_EPOCHS    = 10
PHASE2_EPOCHS    = 10
PHASE2_LR        = 5e-6
UNFREEZE_LAYERS  = 40            # last N layers to unfreeze in phase 2

# Set to a number to train on a subset of classes (e.g. 5 for quick test)
# Set to None to train on all classes
NUM_CLASSES_SUBSET = None

# Clear stale disk cache when config changes (image size / alpha changed)
import shutil
for cache in [CACHE_DIR / "train_cache", CACHE_DIR / "val_cache"]:
    for f in CACHE_DIR.glob(f"{cache.name}*"):
        f.unlink(missing_ok=True)

for d in [MODELS_DIR, OUTPUTS_DIR, CACHE_DIR]:
    d.mkdir(exist_ok=True)

# ─── 1. GPU Check ─────────────────────────────────────────────────────────────
print("=" * 60)
print("  Smart Harvest AI — Disease Model Training")
print("=" * 60)
devices = tf.config.list_physical_devices()
print("\n[GPU] Physical devices detected:")
for d in devices:
    print(f"      {d.device_type}: {d.name}")
metal = [d for d in devices if d.device_type == "GPU"]
if metal:
    print("  ✅ Metal GPU acceleration is ACTIVE")
else:
    print("  ⚠️  No Metal GPU found — training on CPU")
print()

# ─── 2. Load Dataset ──────────────────────────────────────────────────────────
print("[1/6] Loading dataset ...")

# Subset support: symlink or filter not needed — use class_names after load
full_train = tf.keras.utils.image_dataset_from_directory(
    DATASET_DIR,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int"
)
full_val = tf.keras.utils.image_dataset_from_directory(
    DATASET_DIR,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int"
)

all_class_names = full_train.class_names

# Apply subset if configured
if NUM_CLASSES_SUBSET and NUM_CLASSES_SUBSET < len(all_class_names):
    subset_classes = all_class_names[:NUM_CLASSES_SUBSET]
    subset_indices = list(range(NUM_CLASSES_SUBSET))
    print(f"      Subset mode: using {NUM_CLASSES_SUBSET} of {len(all_class_names)} classes")

    def filter_subset(images, labels):
        mask = tf.reduce_any(
            tf.stack([tf.equal(labels, i) for i in subset_indices], axis=1), axis=1
        )
        return images[mask], labels[mask]

    full_train = full_train.map(filter_subset, num_parallel_calls=tf.data.AUTOTUNE).unbatch().batch(BATCH_SIZE)
    full_val   = full_val.map(filter_subset, num_parallel_calls=tf.data.AUTOTUNE).unbatch().batch(BATCH_SIZE)
    class_names = subset_classes
    num_classes = NUM_CLASSES_SUBSET
else:
    class_names = all_class_names
    num_classes = len(class_names)

print(f"      Classes: {num_classes}")
print(f"      Class names: {class_names}")

# ─── 3. Augmentation & Preprocessing ─────────────────────────────────────────
print("\n[2/6] Building preprocessing pipeline ...")

# Compute class weights to handle imbalance
class_counts = {}
for i, cls in enumerate(class_names):
    cls_dir = Path(DATASET_DIR) / cls
    class_counts[i] = len(list(cls_dir.glob("*.*")))
total_samples = sum(class_counts.values())
class_weights = {i: total_samples / (num_classes * c) for i, c in class_counts.items()}
print("      Class weights (top imbalanced):")
for i, w in sorted(class_weights.items(), key=lambda x: -x[1])[:5]:
    print(f"        {class_names[i]:<50s} weight={w:.2f}")

augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal_and_vertical"),
    tf.keras.layers.RandomRotation(0.3),
    tf.keras.layers.RandomZoom(0.2),
    tf.keras.layers.RandomBrightness(0.2),
    tf.keras.layers.RandomContrast(0.2),
], name="augmentation")

rescale = tf.keras.layers.Rescaling(1.0 / 255)

def preprocess_train(images, labels):
    images = augmentation(images, training=True)
    images = rescale(images)
    return images, labels

def preprocess_val(images, labels):
    images = rescale(images)
    return images, labels

# Oversample minority classes so each class has ~target_per_class samples
def oversample(dataset, class_counts, class_names, target_per_class=1800):
    datasets = []
    for i, cls in enumerate(class_names):
        count  = class_counts[i]
        repeat = max(1, round(target_per_class / count))
        cls_ds = (
            dataset.unbatch()
            .filter(lambda img, lbl, idx=i: tf.equal(lbl, idx))
            .repeat(repeat)
        )
        datasets.append(cls_ds)
    balanced = datasets[0]
    for ds in datasets[1:]:
        balanced = balanced.concatenate(ds)
    return balanced.shuffle(5000).batch(BATCH_SIZE)

# Cache raw images to disk first (no augmentation yet) — fast repeated access
# Then apply augmentation AFTER cache so it varies each epoch
train_ds = (
    full_train
    .cache(str(CACHE_DIR / "train_raw"))   # cache raw pixels, not augmented
    .map(preprocess_train, num_parallel_calls=tf.data.AUTOTUNE)
    .shuffle(2000)
    .prefetch(tf.data.AUTOTUNE)
)
val_ds = (
    full_val
    .cache(str(CACHE_DIR / "val_raw"))
    .map(preprocess_val, num_parallel_calls=tf.data.AUTOTUNE)
    .prefetch(tf.data.AUTOTUNE)
)

# ─── 4. Build Model ───────────────────────────────────────────────────────────
print("\n[3/6] Building model ...")

base_model = tf.keras.applications.MobileNetV3Small(
    input_shape=(*IMAGE_SIZE, 3),
    alpha=ALPHA,
    include_top=False,
    weights="imagenet",
    include_preprocessing=False   # we handle rescaling ourselves
)
base_model.trainable = False

inputs  = tf.keras.Input(shape=(*IMAGE_SIZE, 3))
x       = base_model(inputs, training=False)
x       = tf.keras.layers.GlobalAveragePooling2D()(x)
x       = tf.keras.layers.Dense(128, activation="relu")(x)
x       = tf.keras.layers.Dropout(0.3)(x)
outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)

model = tf.keras.Model(inputs, outputs)
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)
model.summary()

# ─── 5. Callbacks ─────────────────────────────────────────────────────────────
callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True, verbose=1),
    tf.keras.callbacks.ModelCheckpoint(CHECKPOINT_PATH, monitor="val_loss", save_best_only=True, verbose=1),
    tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-7, verbose=1),
]

# ─── 6. Phase 1 — Train Head ──────────────────────────────────────────────────
print("\n[4/6] Phase 1 — training head only ...")
print(f"      Epochs: {PHASE1_EPOCHS} | LR: 0.001 | Base: frozen")
print("-" * 60)

history1 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=PHASE1_EPOCHS,
    class_weight=class_weights,
    callbacks=callbacks,
    verbose=1
)

# ─── 7. Phase 2 — Fine-tune ───────────────────────────────────────────────────
print(f"\n[5/6] Phase 2 — unfreezing last {UNFREEZE_LAYERS} layers ...")
print(f"      Epochs: {PHASE2_EPOCHS} | LR: {PHASE2_LR}")
print("-" * 60)

base_model.trainable = True
for layer in base_model.layers[:-UNFREEZE_LAYERS]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=PHASE2_LR),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

history2 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=PHASE2_EPOCHS,
    class_weight=class_weights,
    callbacks=callbacks,
    verbose=1
)

# ─── 8. Evaluate ──────────────────────────────────────────────────────────────
print("\n[6/6] Evaluating ...")
best_model = tf.keras.models.load_model(CHECKPOINT_PATH)
val_loss, val_acc = best_model.evaluate(val_ds, verbose=0)
print(f"\n  ✅ Final val accuracy : {val_acc*100:.2f}%")
print(f"  ✅ Final val loss     : {val_loss:.4f}")

# Per-class accuracy + confusion matrix
all_labels, all_preds = [], []
for images, labels in val_ds:
    preds = best_model.predict(images, verbose=0)
    all_preds.extend(np.argmax(preds, axis=1))
    all_labels.extend(labels.numpy())

all_labels = np.array(all_labels)
all_preds  = np.array(all_preds)

print("\n  Per-class accuracy:")
cm = confusion_matrix(all_labels, all_preds)
for i, cls in enumerate(class_names):
    total   = cm[i].sum()
    correct = cm[i][i]
    pct     = correct / total * 100 if total > 0 else 0
    print(f"    {cls:<50s} {correct:>4}/{total:<4} ({pct:.1f}%)")

# Save confusion matrix
fig, ax = plt.subplots(figsize=(max(10, num_classes), max(8, num_classes - 2)))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
disp.plot(ax=ax, xticks_rotation=45, colorbar=False)
plt.title("Confusion Matrix — Plant Disease Detection")
plt.tight_layout()
cm_path = str(OUTPUTS_DIR / "confusion_matrix.png")
plt.savefig(cm_path, dpi=150)
plt.close()
print(f"\n  📊 Confusion matrix saved → {cm_path}")

# ─── 9. Export TFLite ─────────────────────────────────────────────────────────
print("\n  Converting to TFLite (dynamic range quantization) ...")

# Save as SavedModel first — workaround for MobileNetV3 + TFLite LLVM bug on tensorflow-macos
saved_model_path = str(MODELS_DIR / "saved_model")
best_model.export(saved_model_path)

converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_path)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

with open(TFLITE_PATH, "wb") as f:
    f.write(tflite_model)

size_mb = os.path.getsize(TFLITE_PATH) / (1024 * 1024)
print(f"  📦 TFLite model saved → {TFLITE_PATH}")
print(f"  📏 File size: {size_mb:.2f} MB")
if size_mb > 15:
    print("  ⚠️  Model exceeds 15MB. Consider reducing alpha further (e.g. alpha=0.5).")
else:
    print("  ✅ Under 15MB — safe for Render deployment.")

# ─── 10. Save Class Labels ────────────────────────────────────────────────────
labels_map = {str(i): name for i, name in enumerate(class_names)}
with open(LABELS_PATH, "w") as f:
    json.dump(labels_map, f, indent=2)
print(f"  🏷️  Class labels saved → {LABELS_PATH}")

print("\n" + "=" * 60)
print("  Training complete.")
print(f"  Best model  : {CHECKPOINT_PATH}")
print(f"  TFLite      : {TFLITE_PATH}")
print(f"  Labels      : {LABELS_PATH}")
print(f"  Val accuracy: {val_acc*100:.2f}%")
print("=" * 60)
