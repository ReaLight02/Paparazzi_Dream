'C:/Users/mocan/Desktop/Paparazzi_Dream/Dataset_Train'
'C:/Users/mocan/Desktop/Paparazzi_Dream/Dataset_Test'



import mapping
import data_loader
from keras.models import Sequential
from keras.layers import Input, Conv2D, MaxPool2D, Flatten, Dense
import time

# Creare il modello
classifier = Sequential()

# Aggiungi il livello di input
classifier.add(Input(shape=(64, 64, 3)))

# Aggiungi i livelli di convoluzione e pooling
classifier.add(Conv2D(32, kernel_size=(5, 5), strides=(1, 1), activation='relu'))
classifier.add(MaxPool2D(pool_size=(2, 2)))

# Aggiungi un altro livello di convoluzione e pooling
classifier.add(Conv2D(64, kernel_size=(5, 5), strides=(1, 1), activation='relu'))
classifier.add(MaxPool2D(pool_size=(2, 2)))

# Aggiungi il livello di flattening
classifier.add(Flatten())

# Aggiungi i livelli completamente connessi
classifier.add(Dense(64, activation='relu'))
classifier.add(Dense(mapping.OutputNeurons, activation='softmax'))

# Compilare il modello
classifier.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Misurare il tempo di addestramento
StartTime = time.time()

# Addestrare il modello
classifier.fit(
    data_loader.training_set,
    steps_per_epoch=70,
    epochs=10,
    validation_data=data_loader.test_set,
    validation_steps=15
)

EndTime = time.time()
print("###### Total Time Taken: ", round((EndTime - StartTime) / 60), 'Minutes ######')
