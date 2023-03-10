import os
import librosa
import numpy as np
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical


def load_test_data():
    # Initialize variables to hold audio data and corresponding labels
    X = []
    y = []
    # Get list of audio files in folder
    files = os.listdir(TEST_DATA_PATH)
    for file in files:
        # Load audio file
        audio_path = os.path.join(TEST_DATA_PATH, file)
        audio, sr = librosa.load(audio_path, sr=SAMPLE_RATE)
        # Downsample audio to 2200 Hz
        audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLE_RATE)
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
            y.append(0)  # label is not known for test data
    # Convert data to numpy arrays for return
    X = np.array(X)
    y = np.array(y)
    # Reshape data to match expected input shape
    X = np.transpose(X, (0, 2, 1, 3))
    # Return data and labels
    return X, y
