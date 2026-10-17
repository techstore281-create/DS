"""
Healthcare Data Science Experiment #3
AI for Medical Diagnosis on MRI/X-ray Data
Binary classification: Pneumonia detection from Chest X-rays
Framework: TensorFlow/Keras + MobileNetV2 transfer learning + Grad-CAM
"""

import os, sys, warnings, random, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
import cv2
from PIL import Image
from tqdm import tqdm

import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report, roc_curve
)

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# Reproducibility
SEED = 42
random.seed(SEED); np.random.seed(SEED); tf.random.set_seed(SEED)

os.makedirs("outputs", exist_ok=True)
os.makedirs("data", exist_ok=True)
SEP = "=" * 78

def header(t): print(f"\n{SEP}\n>>> {t}\n{SEP}")
def sub(t):    print(f"\n--- {t} ---")

# Hyperparameters
IMG_SIZE    = 224
BATCH_SIZE  = 16
EPOCHS      = 8          # raise to 20-30 for production
LR          = 1e-4
CLASS_NAMES = ["NORMAL", "PNEUMONIA"]


# =====================================================================
# STEP 1: DATA COLLECTION
# =====================================================================
header("STEP 1: DATA COLLECTION")

REAL_DATA = "data/chest_xray"
SYNTH_DIR = "data/synthetic_xray"

def make_synthetic_split(root, n_per_class, size=(224, 224)):
    """Generate a tiny synthetic X-ray-like dataset as fallback."""
    for split in ["train", "val", "test"]:
        for cls, mu in zip(CLASS_NAMES, [60, 130]):
            d = os.path.join(root, split, cls)
            os.makedirs(d, exist_ok=True)
            n = n_per_class if split == "train" else max(2, n_per_class // 4)
            for i in range(n):
                # Create chest-X-ray-like image (noise + blurred shape)
                base = np.random.normal(mu, 25, size).astype(np.uint8)
                base = cv2.GaussianBlur(base, (15, 15), 0)
                if cls == "PNEUMONIA":  # add "opacity" blobs
                    for _ in range(3):
                        cy = np.random.randint(40, size[0]-40)
                        cx = np.random.randint(40, size[1]-40)
                        cv2.circle(base, (cx, cy), np.random.randint(15, 35),
                                   int(mu + 40), -1)
                    base = cv2.GaussianBlur(base, (21, 21), 0)
                img = np.stack([base]*3, axis=-1)
                cv2.imwrite(os.path.join(d, f"{cls}_{i:04d}.png"), img)


def collect_data():
    if os.path.isdir(os.path.join(REAL_DATA, "train")):
        print(f"[+] Real Chest X-ray dataset found at: {REAL_DATA}")
        root = REAL_DATA
    else:
        print("[!] Real dataset not found.")
        print("[i] To use real data, run:")
        print("    kaggle datasets download -d paultimothymooney/chest-xray-pneumonia")
        print("    unzip chest-xray-pneumonia.zip -d data/ && mv data/chest_xray .")
        print("[+] Generating synthetic fallback dataset...")
        if not os.path.isdir(SYNTH_DIR):
            make_synthetic_split(SYNTH_DIR, n_per_class=40)
        root = SYNTH_DIR

    # Count images per split/class
    print("\n[i] Dataset inventory:")
    for split in ["train", "val", "test"]:
        split_dir = os.path.join(root, split)
        if not os.path.isdir(split_dir):
            print(f"    {split}: MISSING"); continue
        counts = {}
        for cls in sorted(os.listdir(split_dir)):
            cls_dir = os.path.join(split_dir, cls)
            if os.path.isdir(cls_dir):
                counts[cls] = len(os.listdir(cls_dir))
        total = sum(counts.values())
        print(f"    {split:5s}: {counts}  (total={total})")
    return root


DATA_ROOT = collect_data()


# =====================================================================
# STEP 2: DATA CLEANING & PREPROCESSING
# =====================================================================
header("STEP 2: DATA CLEANING & PREPROCESSING")

VALID_EXT = {".png", ".jpg", ".jpeg", ".bmp"}

def clean_dataset(root):
    """Remove corrupt / unreadable / wrong-format files."""
    removed, kept = 0, 0
    for split in ["train", "val", "test"]:
        for cls in CLASS_NAMES:
            d = os.path.join(root, split, cls)
            if not os.path.isdir(d): continue
            for f in os.listdir(d):
                p = os.path.join(d, f)
                ext = os.path.splitext(f)[1].lower()
                if ext not in VALID_EXT:
                    os.remove(p); removed += 1; continue
                try:
                    img = Image.open(p); img.verify()
                    kept += 1
                except Exception:
                    os.remove(p); removed += 1
    print(f"[+] Kept: {kept:5d}   Removed (corrupt/unsupported): {removed}")

clean_dataset(DATA_ROOT)

sub("Class balance check")
for split in ["train", "val", "test"]:
    sd = os.path.join(DATA_ROOT, split)
    if os.path.isdir(sd):
        counts = {c: len(os.listdir(os.path.join(sd, c)))
                  for c in os.listdir(sd)
                  if os.path.isdir(os.path.join(sd, c))}
        total = sum(counts.values())
        print(f"  {split:5s}: ", end="")
        for c, n in counts.items():
            print(f"{c}={n} ({n/total*100:.1f}%)  ", end="")
        print()

sub("Preprocessing config")
print(f"  Image size       : {IMG_SIZE}x{IMG_SIZE}")
print(f"  Batch size       : {BATCH_SIZE}")
print(f"  Normalization    : MobileNetV2 preprocess_input ([-1, 1])")
print(f"  Augmentation     : rotation, zoom, shift, horizontal flip")


# =====================================================================
# STEP 3: DATA INTEGRATION (tf.data pipelines)
# =====================================================================
header("STEP 3: DATA INTEGRATION — tf.data pipelines")

AUTOTUNE = tf.data.AUTOTUNE

def load_and_preprocess(path, label):
    raw = tf.io.read_file(path)
    img = tf.image.decode_image(raw, channels=3, expand_animations=False)
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
    img = tf.cast(img, tf.float32)
    img = preprocess_input(img)
    return img, label

def build_dataset(root, split, shuffle=False, augment=False):
    split_dir = os.path.join(root, split)
    paths, labels = [], []
    for idx, cls in enumerate(sorted(os.listdir(split_dir))):
        cd = os.path.join(split_dir, cls)
        if not os.path.isdir(cd): continue
        for f in os.listdir(cd):
            paths.append(os.path.join(cd, f)); labels.append(idx)

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(paths), seed=SEED)

    ds = ds.map(load_and_preprocess, num_parallel_calls=AUTOTUNE)

    if augment:
        aug = tf.keras.Sequential([
            layers.RandomRotation(0.05),
            layers.RandomZoom(0.10),
            layers.RandomTranslation(0.05, 0.05),
            layers.RandomFlip("horizontal"),
        ])
        ds = ds.map(lambda x, y: (aug(x, training=True), y),
                    num_parallel_calls=AUTOTUNE)

    ds = ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)
    return ds, paths, labels

train_ds, train_paths, train_labels = build_dataset(DATA_ROOT, "train",
                                                    shuffle=True, augment=True)
val_ds,   val_paths,   val_labels   = build_dataset(DATA_ROOT, "val")
test_ds,  test_paths,  test_labels  = build_dataset(DATA_ROOT, "test")

print(f"[+] Train batches: {len(train_ds)}  |  Val: {len(val_ds)}  |  Test: {len(test_ds)}")
print(f"[+] Train images : {len(train_paths)}")
print(f"[+] Val images   : {len(val_paths)}")
print(f"[+] Test images  : {len(test_paths)}")


# =====================================================================
# STEP 4: DATA TRANSFORMATION — Model Architecture
# =====================================================================
header("STEP 4: MODEL — MobileNetV2 Transfer Learning")

def build_model(num_classes=1, fine_tune_at=100):
    base = MobileNetV2(input_shape=(IMG_SIZE, IMG_SIZE, 3),
                       include_top=False, weights="imagenet")
    base.trainable = True
    for layer in base.layers[:fine_tune_at]:
        layer.trainable = False

    inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = base(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(num_classes, activation="sigmoid")(x)

    model = models.Model(inputs, outputs)
    return model, base


model, base_model = build_model()

model.compile(
    optimizer=optimizers.Adam(LR),
    loss="binary_crossentropy",
    metrics=["accuracy", tf.keras.metrics.AUC(name="auc"),
             tf.keras.metrics.Precision(name="precision"),
             tf.keras.metrics.Recall(name="recall")]
)
model.summary()


# =====================================================================
# STEP 5: TRAINING
# =====================================================================
header("STEP 5: MODEL TRAINING")

# Class weights (handle imbalance if present)
from sklearn.utils.class_weight import compute_class_weight
cw = compute_class_weight("balanced", classes=np.unique(train_labels),
                          y=train_labels)
class_weight = {i: w for i, w in enumerate(cw)}
print(f"[+] Class weights: {class_weight}")

ckpt_path = "outputs/best_model.keras"
cbs = [
    callbacks.ModelCheckpoint(ckpt_path, monitor="val_auc",
                              mode="max", save_best_only=True, verbose=1),
    callbacks.EarlyStopping(monitor="val_auc", mode="max",
                            patience=4, restore_best_weights=True),
    callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                patience=2, min_lr=1e-6, verbose=1),
]

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    class_weight=class_weight,
    callbacks=cbs,
    verbose=1,
)


# =====================================================================
# STEP 6: EVALUATION
# =====================================================================
header("STEP 6: EVALUATION ON TEST SET")

print("[+] Loading best checkpoint...")
model = tf.keras.models.load_model(ckpt_path)

y_prob = model.predict(test_ds, verbose=0).ravel()
y_pred = (y_prob >= 0.5).astype(int)
y_true = np.array(test_labels)

metrics = {
    "Accuracy":  accuracy_score(y_true, y_pred),
    "Precision": precision_score(y_true, y_pred, zero_division=0),
    "Recall":    recall_score(y_true, y_pred, zero_division=0),
    "F1-Score":  f1_score(y_true, y_pred, zero_division=0),
    "ROC-AUC":   roc_auc_score(y_true, y_prob) if len(set(y_true)) > 1 else float("nan"),
}
print("\n--- Test Metrics ---")
for k, v in metrics.items():
    print(f"  {k:12s}: {v:.4f}")

print("\n--- Classification Report ---")
print(classification_report(y_true, y_pred, target_names=CLASS_NAMES,
                            zero_division=0))

cm = confusion_matrix(y_true, y_pred)
print("--- Confusion Matrix ---")
print(pd.DataFrame(cm, index=[f"Actual {c}" for c in CLASS_NAMES],
                   columns=[f"Pred {c}" for c in CLASS_NAMES]).to_string())

# Save metrics
with open("outputs/metrics.txt", "w") as f:
    json.dump({k: float(v) for k, v in metrics.items()}, f, indent=2)
print("\n[+] Saved outputs/metrics.txt")


# =====================================================================
# STEP 7: TRAINING CURVES & DIAGNOSTIC PLOTS
# =====================================================================
header("STEP 7: VISUALIZATIONS")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# (a) Loss
axes[0, 0].plot(history.history["loss"], label="train")
axes[0, 0].plot(history.history["val_loss"], label="val")
axes[0, 0].set_title("Loss"); axes[0, 0].set_xlabel("Epoch")
axes[0, 0].legend()

# (b) AUC
axes[0, 1].plot(history.history["auc"], label="train")
axes[0, 1].plot(history.history["val_auc"], label="val")
axes[0, 1].set_title("AUC"); axes[0, 1].set_xlabel("Epoch")
axes[0, 1].legend()

# (c) Confusion Matrix
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[1, 0],
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
axes[1, 0].set_title("Confusion Matrix")
axes[1, 0].set_xlabel("Predicted"); axes[1, 0].set_ylabel("Actual")

# (d) ROC Curve
if len(set(y_true)) > 1:
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    axes[1, 1].plot(fpr, tpr, lw=2,
                    label=f"AUC = {metrics['ROC-AUC']:.3f}")
    axes[1, 1].plot([0, 1], [0, 1], "k--", alpha=0.5)
    axes[1, 1].set_title("ROC Curve")
    axes[1, 1].set_xlabel("FPR"); axes[1, 1].set_ylabel("TPR")
    axes[1, 1].legend()

plt.tight_layout()
plt.savefig("outputs/training_curves.png", dpi=120, bbox_inches="tight")
plt.close()
print("[+] Saved outputs/training_curves.png")


# =====================================================================
# STEP 8: GRAD-CAM EXPLAINABILITY
# =====================================================================
header("STEP 8: GRAD-CAM EXPLAINABILITY")

def grad_cam(model, img_array, base_model_name="mobilenetv2_1.00_224"):
    """
    Compute Grad-CAM heatmap for the last conv layer.
    img_array: preprocessed (1, H, W, 3) tensor.
    """
    base = None
    for l in model.layers:
        if l.name.startswith(base_model_name) or "mobilenet" in l.name:
            base = l; break
    if base is None:
        # fallback: find last conv-containing submodel
        for l in model.layers:
            if hasattr(l, "layers") and len(l.layers) > 10:
                base = l; break

    last_conv = None
    for l in reversed(base.layers):
        if isinstance(l, layers.Conv2D):
            last_conv = l; break

    grad_model = models.Model(
        inputs=model.inputs,
        outputs=[last_conv.output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_out, preds = grad_model(img_array, training=False)
        loss = preds[:, 0]

    grads = tape.gradient(loss, conv_out)
    pooled = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_out = conv_out[0]
    heat = tf.reduce_sum(tf.multiply(pooled, conv_out), axis=-1)
    heat = tf.maximum(heat, 0) / (tf.reduce_max(heat) + 1e-8)
    return heat.numpy(), float(preds[0, 0])


def overlay_gradcam(orig_path, heatmap, alpha=0.45):
    img = cv2.imread(orig_path)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    heat = cv2.resize(heatmap, (IMG_SIZE, IMG_SIZE))
    heat_u8 = np.uint8(255 * heat)
    colored = cv2.applyColorMap(heat_u8, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(img, 1 - alpha, colored, alpha, 0)
    return cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB), cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


# Pick a few test images (one per class if possible)
sample_paths, sample_labels = [], []
seen = set()
for p, l in zip(test_paths, test_labels):
    if l not in seen:
        sample_paths.append(p); sample_labels.append(l); seen.add(l)
    if len(sample_paths) >= 4: break
# top up with extras
for p, l in zip(test_paths, test_labels):
    if len(sample_paths) >= 4: break
    if p not in sample_paths:
        sample_paths.append(p); sample_labels.append(l)

fig, axes = plt.subplots(len(sample_paths), 3,
                         figsize=(10, 3.2 * len(sample_paths)))
if len(sample_paths) == 1: axes = axes.reshape(1, -1)

for i, (p, true_l) in enumerate(zip(sample_paths, sample_labels)):
    # Preprocess
    raw = tf.io.read_file(p)
    img = tf.image.decode_image(raw, channels=3)
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
    arr = preprocess_input(tf.cast(img, tf.float32).numpy()[None, ...])

    heat, prob = grad_cam(model, tf.constant(arr))
    pred_l = 1 if prob >= 0.5 else 0
    overlay, original = overlay_gradcam(p, heat)

    axes[i, 0].imshow(original); axes[i, 0].axis("off")
    axes[i, 0].set_title(f"Original\nTrue: {CLASS_NAMES[true_l]}", fontsize=9)

    axes[i, 1].imshow(heat, cmap="jet"); axes[i, 1].axis("off")
    axes[i, 1].set_title("Grad-CAM heatmap", fontsize=9)

    axes[i, 2].imshow(overlay); axes[i, 2].axis("off")
    ok = "✓" if pred_l == true_l else "✗"
    axes[i, 2].set_title(
        f"Overlay {ok}\nPred: {CLASS_NAMES[pred_l]} (p={prob:.3f})", fontsize=9)

plt.suptitle("Grad-CAM: Model Attention on Chest X-rays", fontsize=13)
plt.tight_layout()
plt.savefig("outputs/gradcam_examples.png", dpi=120, bbox_inches="tight")
plt.close()
print("[+] Saved outputs/gradcam_examples.png")

# Print numeric attention summary
print("\n[i] Grad-CAM focus report:")
for p, true_l in zip(sample_paths, sample_labels):
    raw = tf.io.read_file(p); img = tf.image.decode_image(raw, channels=3)
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
    arr = preprocess_input(tf.cast(img, tf.float32).numpy()[None, ...])
    heat, prob = grad_cam(model, tf.constant(arr))
    hot_pixels = (heat > 0.5).mean() * 100
    print(f"  {os.path.basename(p):20s}  true={CLASS_NAMES[true_l]:9s}  "
          f"p(pneumonia)={prob:.3f}  focus={hot_pixels:.1f}% of image")


# =====================================================================
# STEP 9: MRI ADAPTER NOTE
# =====================================================================
header("STEP 9: MRI ADAPTER (multi-class)")

print("""
[i] To adapt this pipeline for MRI (e.g., brain tumor classification):

    1. Dataset:
       kaggle datasets download -d masoudnickparvar/brain-tumor-mri-dataset
       # Classes: glioma, meningioma, pituitary, notumor (4 classes)

    2. Code changes (3 lines):
       NUM_CLASSES = 4
       outputs = layers.Dense(NUM_CLASSES, activation='softmax')(x)
       loss    = 'categorical_crossentropy'
       metrics = ['accuracy', tf.keras.metrics.AUC(name='auc')]
       CLASS_NAMES = ['glioma', 'meningioma', 'pituitary', 'notumor']

    3. Re-run — everything else (data loading, Grad-CAM, evaluation,
       visualization) works unchanged.
""")


# =====================================================================
# STEP 10: SUMMARY
# =====================================================================
header("PIPELINE COMPLETE ✅")

print(f"Dataset root      : {DATA_ROOT}")
print(f"Train / Val / Test: {len(train_paths)} / {len(val_paths)} / {len(test_paths)}")
print(f"Model             : MobileNetV2 + custom head (fine-tuned)")
print(f"Best checkpoint   : {ckpt_path}")
print(f"\nTest metrics:")
for k, v in metrics.items():
    print(f"  {k:12s}: {v:.4f}")

print("\nOutputs saved to ./outputs/:")
for f in sorted(os.listdir("outputs")):
    print(f"  - {f}")

print(SEP)