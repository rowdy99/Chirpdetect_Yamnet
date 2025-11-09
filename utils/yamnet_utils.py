# utils/yamnet_utils.py
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import os
import shutil

_yamnet_model = None

def load_yamnet():
    global _yamnet_model
    if _yamnet_model is None:
        try:
            _yamnet_model = hub.load("https://tfhub.dev/google/yamnet/1")
        except ValueError as e:
            if "incompatible/unknown type" in str(e):
                # Clear corrupted cache and retry
                cache_dir = os.path.join(os.environ.get('TEMP', '/tmp'), 'tfhub_modules')
                if os.path.exists(cache_dir):
                    shutil.rmtree(cache_dir)
                _yamnet_model = hub.load("https://tfhub.dev/google/yamnet/1")
            else:
                raise
    return _yamnet_model

def waveform_to_yamnet_embeddings(waveform, sr=16000):
    """
    waveform: 1D numpy float32 array at sr (should be 16000)
    returns: averaged embedding vector (1024)
    """
    model = load_yamnet()
    # model expects float32 mono waveform, sample rate 16000
    waveform = waveform.astype(np.float32)
    # model signature: scores, embeddings, spectrogram
    scores, embeddings, spectrogram = model(waveform)
    # embeddings shape: (num_patches, 1024)
    embeddings_np = embeddings.numpy()
    # average across time/patches -> single vector
    avg = np.mean(embeddings_np, axis=0)
    return avg
