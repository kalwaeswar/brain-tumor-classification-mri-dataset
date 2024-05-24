import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

# Load the model
model = load_model("my_model.h5")

# Define class names
class_names = ['glioma', 'meningioma', 'notumor', 'pituitary']

# Preprocess the uploaded image
def preprocess_image(image):
    # Resize the image to match the input shape of the model
    image = image.resize((130, 130))
    
    # Convert the image to a numpy array
    image_array = np.array(image)
    
    # Ensure the image has 3 channels (RGB)
    if len(image_array.shape) == 2:
        image_array = np.stack((image_array,) * 3, axis=-1)
    
    # Normalize the pixel values to be in the range [0, 1]
    image_array = image_array / 255.0
    
    # Add batch dimension
    image_array = np.expand_dims(image_array, axis=0)
    
    return image_array

# Streamlit app
st.title("Image Classification")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Make predictions
    if st.button('Classify'):
        st.write("Predicting...")
        processed_image = preprocess_image(image)
        prediction = model.predict(processed_image)
        class_index = np.argmax(prediction[0])
        st.write(f"Prediction: {class_names[class_index]}")
