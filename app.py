import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet import preprocess_input

# Load the trained model (make sure the file is in the same directory or adjust the path)
model = load_model("waste_classifier_mobilenet.h5")
# Define the class names as used during training
class_names = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]

st.title("Waste Classification App")
st.write("Upload an image of waste and let the model predict its category.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Convert the file data to a NumPy array and then decode to an image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    
    # Show the uploaded image
    st.image(image, channels="BGR", caption="Uploaded Image")
    
    # Preprocess the image: resize and apply MobileNet preprocessing
    image_resized = cv2.resize(image, (224, 224))
    image_processed = preprocess_input(np.array(image_resized, dtype=np.float32))
    image_processed = np.expand_dims(image_processed, axis=0)
    
    # Predict the waste category
    prediction = model.predict(image_processed)
    predicted_index = np.argmax(prediction, axis=1)[0]
    predicted_category = class_names[predicted_index]
    
    st.write(f"**Predicted Category:** {predicted_category}")