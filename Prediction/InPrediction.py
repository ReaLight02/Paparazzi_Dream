import cv2
import numpy as np
from filters import Filters
from keras.preprocessing import image
from cropping import get_cropped_image_2_eyes
from keras.models import load_model


# Load the trained model
model = load_model("model.h5")


def preprocess_image(img_array):
    # Transform into image, resize, transform back into numpy array, expand dimension and normalize values
    img = image.array_to_img(img_array, data_format=None, scale=True)
    img = img.resize((64, 64))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return img_array


# Classify the input image
def classify_image(model, image_path):
    processed_image = preprocess_image(image_path)
    # Use model to predict based on the processed image
    prediction = model.predict(processed_image)
    return prediction


# Interpret the prediction
class_names = [
    "Angelina Jolie",
    "Brad Pitt",
    "Denzel Washington",
    "Hugh Jackman",
    "Jennifer Lawrence",
    "Johnny Depp",
    "Kate Winslet",
    "Leonardo DiCaprio",
    "Megan Fox",
    "Natalie Portman",
    "Nicole Kidman",
    "Robert Downey Jr",
    "Sandra Bullock",
    "Scarlett Johansson",
    "Tom Cruise",
    "Tom Hanks",
    "Will Smith",
]


def interpret_prediction(prediction):
    predicted_class_index = np.argmax(prediction)
    predicted_class_name = class_names[predicted_class_index]
    return predicted_class_name


if __name__ == "__main__":
    image_path = r"E:\LAB\jol.jpg"

    # Apply glasses and mustache filters
    face_image = cv2.imread(image_path)
    Filters(face_image)

    # Crop the image
    cropped = get_cropped_image_2_eyes(image_path)
    # Convert BGR to RGB
    cropped = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)

    # Classify the image and interpret the prediction
    prediction = classify_image(model, cropped)
    predicted_class_name = interpret_prediction(prediction)
    print(f"Prediction: {predicted_class_name}")

