import tensorflow as tf
import numpy as np
from PIL import Image

IMAGE_SIZE = 224
MODEL_PATH = "model/AI_plant_diseases_detection (1).keras"

# Load Keras model
model = tf.keras.models.load_model(MODEL_PATH)

class_names = [
    'Apple___Apple_scab',
    'Apple___Black_rot',
    'Apple___Cedar_apple_rust',
    'Apple___healthy',
    'Blueberry___healthy',
    'Cherry___Powdery_mildew',
    'Cherry___healthy',
    'Corn___Cercospora_leaf_spot',
    'Corn___Common_rust',
    'Corn___Northern_Leaf_Blight',
    'Corn___healthy',
    'Grape___Black_rot',
    'Grape___Esca',
    'Grape___Leaf_blight',
    'Grape___healthy',
    'Orange___Haunglongbing',
    'Peach___Bacterial_spot',
    'Peach___healthy',
    'Pepper___Bacterial_spot',
    'Pepper___healthy',
    'Potato___Early_blight',
    'Potato___Late_blight',
    'Potato___healthy',
    'Raspberry___healthy',
    'Soybean___healthy',
    'Squash___Powdery_mildew',
    'Strawberry___Leaf_scorch',
    'Strawberry___healthy',
    'Tomato___Bacterial_spot',
    'Tomato___Early_blight',
    'Tomato___Late_blight',
    'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot',
    'Tomato___Spider_mites',
    'Tomato___Target_Spot',
    'Tomato___Yellow_Leaf_Curl_Virus',
    'Tomato___Tomato_mosaic_virus',
    'Tomato___healthy'
]

def preprocess_image(image: Image.Image):
    image = image.resize((IMAGE_SIZE, IMAGE_SIZE))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

def predict(image: Image.Image):
    img = preprocess_image(image)
    predictions = model.predict(img)

    idx = np.argmax(predictions[0])
    confidence = float(predictions[0][idx])

    return {
        "label": class_names[idx],
        "confidence": confidence,
        "description": "Plant disease detected using AI model.",
        "remedy": "Consult an agricultural expert or apply recommended treatment."
    }
