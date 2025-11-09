# utils/preprocess.py
import librosa
import numpy as np
import random
import matplotlib.pyplot as plt
import librosa.display
import os
from .yamnet_utils import waveform_to_yamnet_embeddings

def save_spectrogram(y, sr, filename, save_dir="spectrograms", n_mels=128):
    """
    Generates and saves a Mel spectrogram PNG for visualization/report.
    """
    os.makedirs(save_dir, exist_ok=True)
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

    plt.figure(figsize=(4, 4))
    librosa.display.specshow(mel_spec_db, sr=sr, x_axis='time', y_axis='mel', cmap='magma')
    plt.axis('off')
    plt.tight_layout(pad=0)

    save_path = os.path.join(save_dir, f"{os.path.splitext(os.path.basename(filename))[0]}_spec.png")
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close()
    return save_path

def load_audio_fixed(path, sr=16000, duration=3.0):

    try:
        y, _sr = librosa.load(path, sr=sr, mono=True, duration=None)
    except Exception as e:
        raise IOError(f"Could not load {path}: {e}")

    target_len = int(sr * duration)
    if len(y) < target_len:
        y = np.pad(y, (0, target_len - len(y)))
    else:
        y = y[:target_len]
    return y, sr

def audio_to_melspec(y, sr, n_mels=128, n_fft=1024, hop_length=256, power=2.0, normalize=True):
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels,
                                       n_fft=n_fft, hop_length=hop_length, power=power)
    S_db = librosa.power_to_db(S, ref=np.max)
    if normalize:
        S_db = (S_db - S_db.min()) / (S_db.max() - S_db.min() + 1e-8)
    return S_db.astype(np.float32)

# ------- Augmentation helpers -------
def add_noise(y, noise_factor=0.005):
    noise = np.random.randn(len(y))
    return y + noise_factor * noise

def time_stretch(y, rate=1.0):
    try:
        y_st = librosa.effects.time_stretch(y, rate)
    except Exception:
        return y
    if len(y_st) < len(y):
        y_st = np.pad(y_st, (0, len(y)-len(y_st)))
    else:
        y_st = y_st[:len(y)]
    return y_st

def pitch_shift(y, sr, n_steps):
    try:
        return librosa.effects.pitch_shift(y, sr, n_steps=n_steps)
    except Exception:
        return y

def random_augment(y, sr):
    aug = y.copy()
    if random.random() < 0.5:
        aug = add_noise(aug, noise_factor=random.uniform(0.002, 0.01))
    if random.random() < 0.4:
        aug = pitch_shift(aug, sr, n_steps=random.randint(-2, 2))
    if random.random() < 0.3:
        aug = time_stretch(aug, rate=random.uniform(0.85, 1.15))
    if len(aug) != len(y):
        if len(aug) < len(y):
            aug = np.pad(aug, (0, len(y)-len(aug)))
        else:
            aug = aug[:len(y)]
    return aug

def waveform_to_embedding(y, sr=16000):
    """
    Wrapper to produce a fixed-size embedding vector using YAMNet.
    """
    return waveform_to_yamnet_embeddings(y, sr=sr)
