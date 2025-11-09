
import streamlit as st
import numpy as np
import tensorflow as tf
import librosa
import os
import matplotlib.pyplot as plt
import librosa.display
from utils.yamnet_utils import load_yamnet, waveform_to_yamnet_embeddings

MODEL_PATH = "model/bird_yamnet_classifier.h5"
LABELS_PATH = "model/labels.npy"
SR = 16000
CLIP_DURATION = 3.0  # seconds

st.title("🎶 ChirpDetect - Bird Sound Classifier (YAMNet embeddings)")
st.write("Upload a bird sound (.wav or .mp3) to identify species and count occurrences (per 3s clip).")

uploaded_file = st.file_uploader("Upload Audio", type=["wav", "mp3"])

@st.cache_resource
def load_classifier(path):
    if not os.path.exists(path):
        st.error("Classifier model not found. Train the model first and save to model/bird_yamnet_classifier.h5")
        return None
    model = tf.keras.models.load_model(path)
    return model

@st.cache_resource
def load_labels(path):
    if not os.path.exists(path):
        return None
    return np.load(path)

classifier = load_classifier(MODEL_PATH)
labels = load_labels(LABELS_PATH)

if uploaded_file and classifier is not None and labels is not None:
    # Load audio directly from uploaded file (supports both .wav and .mp3)
    try:
        y, sr = librosa.load(uploaded_file, sr=SR, mono=True)
    except Exception as e:
        st.error(f"Error loading audio file: {e}")
        st.stop()

    # Generate and display spectrogram for uploaded file
    fig, ax = plt.subplots(figsize=(6, 4))
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    S_db = librosa.power_to_db(S, ref=np.max)
    librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='mel', cmap='magma', ax=ax)
    ax.set_title("Mel Spectrogram")
    ax.axis('off')
    st.pyplot(fig)

    clip_samples = int(CLIP_DURATION * SR)
    num_clips = max(1, len(y) // clip_samples)
    counts = {lbl: 0 for lbl in labels}

    for i in range(num_clips):
        clip = y[i*clip_samples:(i+1)*clip_samples]
        if len(clip) < clip_samples:
            clip = np.pad(clip, (0, clip_samples - len(clip)))
        # Get embeddings
        try:
            emb = waveform_to_yamnet_embeddings(clip, sr=SR)  # (1024,)
        except Exception as e:
            st.error(f"Error generating embeddings for clip {i+1}: {e}")
            continue
        probs = classifier.predict(emb.reshape(1, -1), verbose=0)[0]
        pred_idx = int(np.argmax(probs))
        pred_label = labels[pred_idx]
        counts[pred_label] += 1

    st.subheader("Prediction Results (per 3s clip)")
    for species, count in counts.items():
        if count > 0:
            st.write(f"🐦 {species}: {count} occurrence(s)")

    max_species = max(counts, key=counts.get)
    st.success(f"✅ Most Frequent Bird: **{max_species}** ({counts[max_species]} times)")
elif uploaded_file:
    st.error("Model or labels not found. Train the model first.")