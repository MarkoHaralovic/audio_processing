import os
import numpy as np
from tensorflow.keras.utils import to_categorical
from my_model import get_conv_model, predict
from loading_training_data import load_data
from loading_test_data import load_test_data

# Define constants
DATA_PATH = 'C:\LumenDataScience\Datasets\Dataset\IRMAS_Training_Data'
TEST_DATA_PATH = 'C:\\LumenDataScience\\Datasets\\VALIDATION'
SAMPLE_RATE = 2200
DURATION = 1.0
NUM_MFCC = 2200
N_CLASSES = 11

# Load data
X_train, y_train, X_val, y_val = load_data(split_ratio=0.2)

# Define the model
model = get_conv_model()

# Train the model
history = model.fit(X_train, y_train, validation_data=(X_val, y_val),
                    epochs=100, batch_size=32)


# Load test data
X_test, y_test = load_test_data()

# Make predictions on test data
predicted_instruments = predict(model, X_test)

# Print predicted instrument labels for each audio file
for i, file in enumerate(os.listdir(TEST_DATA_PATH)):
    print(f"File: {file} - Predicted Instrument: {predicted_instruments[i]}")

# Save the model and weights
model.save('my_model.h5')
model.save_weights('my_weights.h5')
