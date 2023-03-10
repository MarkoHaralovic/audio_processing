import os
import librosa
import numpy as np
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from tensorflow.keras.regularizers import l2

# Define constants
DATA_PATH = 'C:\\LumenDataScience\\Datasets\\Dataset\\IRMAS_Training_Data'
SAMPLE_RATE = 2200  # 2200 Hz
DURATION = 1.0  # duration of audio segments in seconds
NUM_MFCC = 13  # number of Mel frequency cepstral coefficients to extract
N_CLASSES = 11

# Load audio files and segment them into one-second intervals


def load_data():
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
            # Downsample audio to 22050 Hz
            audio = librosa.resample(audio, orig_sr=sr,  target_sr=SAMPLE_RATE)
            # Segment audio into one-second intervals
            for segment in range(int(len(audio) / (SAMPLE_RATE * DURATION))):
                start = int(segment * SAMPLE_RATE * DURATION)
                end = int(start + SAMPLE_RATE * DURATION)
                # Extract Mel spectrogram
                mfccs = librosa.feature.mfcc(
                    y=audio[start:end], sr=SAMPLE_RATE, n_mfcc=NUM_MFCC)
                X.append(mfccs)
                y.append(i)  # label is index of instrument folder
    # Convert labels to one-hot encoding
    le = LabelEncoder()
    y = to_categorical(le.fit_transform(y))
    # Convert data to numpy arrays
    X = np.array(X)
    y = np.array(y)
    # Return data and labels
    return X, y

# Define the model


def get_conv_model(N_CLASSES=N_CLASSES, SR=SAMPLE_RATE, DT=DURATION,  NUM_MELS=128):
    input_shape = (int(SR * DT), NUM_MELS, 1)
    model = Sequential()

    # Step IV: Generate Mel Spectrogram
    model.add(Conv2D(32, (3, 3), activation='relu',
              padding='same', input_shape=input_shape))
    model.add(MaxPooling2D((3, 3), padding='same'))
    model.add(Dropout(0.25))

    # Step V: Add Convolutional Layers
    num_filters = 32
    while num_filters != 512:
        model.add(Conv2D(num_filters, (3, 3), activation='relu', padding='same'))
        model.add(MaxPooling2D((3, 3), padding='same'))
        model.add(Dropout(0.25))
        num_filters *= 2

    # Step VI: Add Flatten Layer
    model.add(Flatten())

    # Step VII: Add Fully Connected Layer
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))

    # Step IX: Use Softmax function for calculation of the posterior probability
    model.add(Dense(N_CLASSES, activation='softmax'))

    # Compile the model
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model


# Load data
X, y = load_data()

# Train the model
model = get_conv_model()
history = model.fit(X, y, validation_split=0.2, epochs=50, batch_size=32)
