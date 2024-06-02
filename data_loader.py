from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Definisci i trasformatori di pre-elaborazione per le immagini di addestramento
# Aggiungi trasformazioni per ruotare, zoomare e capovolgere orizzontalmente le immagini
TrainingImagePath=TrainingImagePath = 'C:/Users/mocan/Desktop/Paparazzi_Dream/cropped/'
train_datagen = ImageDataGenerator(
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True
)

# Definisci il trasformatore di pre-elaborazione per le immagini di test (senza data augmentation)
test_datagen = ImageDataGenerator()

# Genera dati di addestramento
training_set = train_datagen.flow_from_directory(
    TrainingImagePath,
    target_size=(64, 64),
    batch_size=32,
    class_mode='categorical'
)

# Genera dati di test
test_set = test_datagen.flow_from_directory(
    TrainingImagePath,
    target_size=(64, 64),
    batch_size=32,
    class_mode='categorical'
)
