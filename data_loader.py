from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Definisci i trasformatori di pre-elaborazione per le immagini di addestramento
# Aggiungi trasformazioni per ruotare, zoomare e capovolgere orizzontalmente le immagini
TrainingImagePath=TrainingImagePath = 'E:/LAB/cropped/'
train_datagen = ImageDataGenerator( 
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    rescale=1./255,
)

# Definisci il trasformatore di pre-elaborazione per le immagini di test
test_datagen = ImageDataGenerator(validation_split=0.2, rescale=1./255,)

# Genera dati di addestramento
training_set = train_datagen.flow_from_directory(
    TrainingImagePath,
    target_size=(64, 64),
    batch_size=64,
    subset = 'training',
    class_mode='categorical'
)

# Genera dati di test
test_set = test_datagen.flow_from_directory(
    TrainingImagePath,
    target_size=(64, 64),
    batch_size=64,
    subset = 'validation',
    class_mode='categorical'
)
