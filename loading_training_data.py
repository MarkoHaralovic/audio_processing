import os
import librosa
import numpy as np
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
import pandas as pd

# Define constants
DATA_PATH = 'C:\LumenDataScience\Datasets\Dataset\IRMAS_Training_Data'
SAMPLE_RATE = 2200
DURATION = 1.0
NUM_MFCC = 2200
N_CLASSES = 11

# Define envelope function


def envelope(y, rate, threshold):
    mask = []
    y = pd.Series(y).apply(np.abs)
    y_mean = y.rolling(window=int(rate/20),
                       min_periods=1,
                       center=True).max()
    for mean in y_mean:
        if mean > threshold:
            mask.append(True)
        else:
            mask.append(False)
    return mask, y_mean

# Load audio files and segment them into one-second intervals


def load_data(split_ratio=0.2, threshold=0.01):
    # Get list of instrument folders
    instrument_folders = sorted(os.listdir(DATA_PATH))
    # Initialize variables to hold audio data and corresponding labels
    X = []
    y = []
    for i, folder in enumerate(instrument_folders):
        # Get list of audio files in folder
        files = os.listdir(os.path.join(DATA_PATH, folder))
        for file in files:
            # Load audio file
            audio_path = os.path.join(DATA_PATH, folder, file)
            audio, sr = librosa.load(audio_path, sr=SAMPLE_RATE)
            # Downsample audio to 2200 Hz
            audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLE_RATE)
            # Apply envelope function to audio file
            mask, env = envelope(audio, sr, threshold)
            audio = audio[mask]
            # Segment audio into one-second intervals
            for segment in range(int(len(audio) / (SAMPLE_RATE * DURATION))):
                start = int(segment * SAMPLE_RATE * DURATION)
                end = int(start + SAMPLE_RATE * DURATION)
                # Extract Mel spectrogram
                mfccs = librosa.feature.mfcc(
                    y=audio[start:end], sr=SAMPLE_RATE, n_mfcc=NUM_MFCC)
                # Resize Mel spectrogram to expected shape
                mfccs = np.resize(
                    mfccs, (int(DURATION * SAMPLE_RATE / 128), NUM_MFCC, 1))
                X.append(mfccs)
                y.append(i)  # label is index of instrument folder
    # Convert labels to one-hot encoding
    le = LabelEncoder()
    y = to_categorical(le.fit_transform(y))
    # Convert data to numpy arrays
    X = np.array(X)
    y = np.array(y)
    # Split data into training and validation sets
    n_samples = X.shape[0]
    n_val_samples = int(split_ratio * n_samples)
    val_indices = np.random.choice(
        range(n_samples), size=n_val_samples, replace=False)
    train_indices = np.delete(range(n_samples), val_indices)
    X_train, y_train = X[train_indices], y[train_indices]
    X_val, y_val = X[val_indices], y[val_indices]
    # Reshape data to match expected input shape
    X_train = np.transpose(X_train, (0, 2, 1, 3))
    X_val = np.transpose(X_val, (0, 2, 1, 3))
    # Return data and labels
    return X_train, y_train, X_val, y_val
