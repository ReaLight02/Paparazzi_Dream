from tensorflow.keras.preprocessing.image import ImageDataGenerator


TrainingImagePath = "E:/LAB/cropped/"

# Data augmentation and normalization
train_datagen = ImageDataGenerator(
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    rescale=1.0 / 255,
    validation_split=0.2,
)

# Create training data
training_set = train_datagen.flow_from_directory(
    TrainingImagePath,
    target_size=(64, 64),
    batch_size=32,
    subset="training",
    class_mode="categorical",
    color_mode="rgb",
)

# Create validation data
test_set = train_datagen.flow_from_directory(
    TrainingImagePath,
    target_size=(64, 64),
    batch_size=32,
    subset="validation",
    class_mode="categorical",
    color_mode="rgb",
)
