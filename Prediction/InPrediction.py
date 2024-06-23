from keras.models import load_model
from keras.preprocessing import image
import numpy as np
from cropping import get_cropped_image_2_eyes
import cv2

# Load the trained model
model = load_model('fizzio.h5')

def preprocess_image(img_array):
    # Assuming img_array is already in the correct format (e.g., uint8 or float32)
    img = image.array_to_img(img_array, data_format=None, scale=True)
    img = img.resize((64, 64))  # Resize to the target size (64x64)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array /= 255.0  # Scale pixel values to [0, 1] range if rescale was used during training
    return img_array

# Classify the input image
def classify_image(model, image_path):
    processed_image = preprocess_image(image_path)
    prediction = model.predict(processed_image)
    return prediction

# Interpret the prediction
class_names = ['Angelina Jolie', 'Brad Pitt', 'Denzel Washington', 'Hugh Jackman', 'Jennifer Lawrence', 'Johnny Depp', 'Kate Winslet', 'Leonardo DiCaprio', 'Megan Fox', 'Natalie Portman', 'Nicole Kidman', 'Robert Downey Jr', 'Sandra Bullock', 'Scarlett Johansson', 'Tom Cruise', 'Tom Hanks', 'Will Smith']

def interpret_prediction(prediction):
    predicted_class_index = np.argmax(prediction)
    predicted_class_name = class_names[predicted_class_index]
    return predicted_class_name

# Path to the input image
image_path = 'E:\LAB/smith.jpg'

cropped = get_cropped_image_2_eyes(image_path)
cropped = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB if needed

# Classify the image and interpret the prediction
prediction = classify_image(model, cropped)

predicted_class_name = interpret_prediction(prediction)
    
print(f"The predicted class for the input image is: {predicted_class_name}")
