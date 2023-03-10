import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense


# path to the folder with trainign data
DATA_PATH = 'C:\LumenDataScience\Datasets\Dataset\IRMAS_Training_Data'
# path to the folder with validation data
TEST_DATA_PATH = 'C:\\LumenDataScience\\Datasets\\VALIDATION'


def get_conv_model(N_CLASSES, SR, DT, NUM_MELS):
    input_shape = (int(SR * DT), NUM_MELS, 1)
    model = Sequential()

    # Step IV: Generate Mel Spectrogram -> step IV. as in pseudo  code from the paper
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


def predict(model, X_test):
    # Make predictions on test data
    y_pred = model.predict(X_test)
    # Convert predictions to instrument labels
    le = LabelEncoder()
    le.fit(os.listdir(DATA_PATH))
    instrument_labels = le.inverse_transform(np.argmax(y_pred, axis=1))
    return instrument_labels
