import time
import mapping
import data_loader
from keras.models import Sequential
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.layers import Input, Conv2D, MaxPool2D, Flatten, Dense, Dropout
from tensorflow.keras.regularizers import l2


# Create model
classifier = Sequential()

# Add input layer
classifier.add(Input(shape=(64, 64, 3)))

classifier.add(Conv2D(32, (5, 5), activation="relu", kernel_regularizer=l2(0.001)))
classifier.add(MaxPool2D(pool_size=(2, 2)))
classifier.add(Dropout(0.25))

classifier.add(Conv2D(64, (3, 3), activation="relu", kernel_regularizer=l2(0.001)))
classifier.add(MaxPool2D(pool_size=(2, 2)))
classifier.add(Dropout(0.25))

# Add flatten layer
classifier.add(Flatten())

# Add fully connected layer
classifier.add(Dense(128, activation="relu", kernel_regularizer=l2(0.001)))
classifier.add(Dense(mapping.OutputNeurons, activation="softmax"))

# Compile the model
classifier.compile(
    loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"]
)

# Save time for later use
StartTime = time.time()

# Define early stopping callback
early_stopping = EarlyStopping(
    monitor="val_loss", patience=10, min_delta=1e-4, verbose=1
)
rlronp = ReduceLROnPlateau(monitor="val_loss", factor=0.8, patience=5, verbose=1)

# Model training and evaluation
classifier.fit(
    data_loader.training_set,
    epochs=100,
    batch_size=32,
    callbacks=[early_stopping, rlronp],
    validation_data=data_loader.test_set,
)

EndTime = time.time()

print("###### Total Time Taken: ", round((EndTime - StartTime) / 60), "Minutes ######")

# Save cnn model
classifier.save("model.h5")
