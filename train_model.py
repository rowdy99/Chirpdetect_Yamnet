import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from utils.preprocess import load_audio_fixed, random_augment, waveform_to_embedding, save_spectrogram

# ---------------- CONFIG ----------------
DATA_DIR = "Data/Raw_audio"
MODEL_DIR = "model"
MODEL_PATH = os.path.join(MODEL_DIR, "bird_yamnet_classifier.h5")
LABELS_PATH = os.path.join(MODEL_DIR, "labels.npy")

SR = 16000
DURATION = 3.0       # seconds per sample
AUGMENT_PER_SAMPLE = 2

os.makedirs(MODEL_DIR, exist_ok=True)

# ---------------- DATA PREPARATION ----------------
def gather_embeddings():
    X_list = []
    y_list = []
    class_names = sorted([d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))])
    if not class_names:
        raise SystemExit("No class directories found in Data/Raw_audio. Put each species in its own folder.")
    print("Classes found:", class_names)

    for idx, cls in enumerate(class_names):
        cls_folder = os.path.join(DATA_DIR, cls)
        files = [f for f in os.listdir(cls_folder) if f.lower().endswith(('.wav', '.mp3', '.flac', '.ogg'))]
        for f in files:
            path = os.path.join(cls_folder, f)
            try:
                y, sr = load_audio_fixed(path, sr=SR, duration=DURATION)
                save_spectrogram(y, sr, path)   # Save spectrogram for visual dataset record
            except Exception as e:
                print("Skipping file (load error):", path, e)
                continue

            variants = [y]
            for _ in range(AUGMENT_PER_SAMPLE):
                variants.append(random_augment(y, SR))

            for waveform in variants:
                try:
                    emb = waveform_to_embedding(waveform, sr=SR)  # (1024,)
                except Exception as e:
                    print("Skipping embedding error:", path, e)
                    continue
                X_list.append(emb)
                y_list.append(idx)

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int32)
    return X, y, class_names

# ---------------- MODEL ----------------
def build_classifier(input_dim, num_classes):
    inp = layers.Input(shape=(input_dim,))
    x = layers.Dense(512, activation='relu')(inp)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    out = layers.Dense(num_classes, activation='softmax')(x)
    model = models.Model(inputs=inp, outputs=out)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

# ---------------- MAIN TRAINING ----------------
if __name__ == "__main__":
    print("Gathering embeddings (this may take a while on first run because YAMNet will be downloaded)...")
    X, y, class_names = gather_embeddings()
    print("X shape:", X.shape, "y shape:", y.shape)
    if len(X) == 0:
        raise SystemExit("No samples found. Check DATA_DIR and audio files.")

    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Compute class weights
    classes = np.unique(y_train)
    class_weights_values = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
    class_weights = dict(enumerate(class_weights_values))
    print("Class weights:", class_weights)

    # Build and compile model
    model = build_classifier(input_dim=X_train.shape[1], num_classes=len(class_names))
    model.summary()

    # Callbacks
    checkpoint = ModelCheckpoint(MODEL_PATH, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
    early = EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-7)

    # Train model
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=50,
        batch_size=32,
        class_weight=class_weights,
        callbacks=[checkpoint, early, reduce_lr]
    )

    # ---------------- METRICS & PLOTS ----------------
    plt.figure(figsize=(8, 5))
    plt.plot(history.history['accuracy'], color='orange', label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], color='skyblue', label='Validation Accuracy')
    plt.title('Accuracy Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.savefig('model/accuracy_curve.png')
    plt.show()

    plt.figure(figsize=(8, 5))
    plt.plot(history.history['loss'], color='red', label='Train Loss')
    plt.plot(history.history['val_loss'], color='blue', label='Validation Loss')
    plt.title('Loss Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('model/loss_curve.png')
    plt.show()

    # Final metrics
    final_train_acc = history.history['accuracy'][-1]
    final_val_acc = history.history['val_accuracy'][-1]
    print(f"\nFinal Training Accuracy: {final_train_acc:.4f}")
    print(f"Final Validation Accuracy: {final_val_acc:.4f}")

    # Save labels
    np.save(LABELS_PATH, np.array(class_names))
    print("\n✅ Training finished successfully.")
    print(f"Model saved at: {MODEL_PATH}")
    print(f"Accuracy and loss plots saved in: {MODEL_DIR}")
