import streamlit as st
import numpy as np
from keras.models import load_model
from PIL import Image
import io

# Load the pre-trained model
model = load_model('uniform.h5')  # Ensure you have your trained model saved

# Streamlit app
st.title("Uniform Detection")
st.write("Upload an image of the person wearing a uniform to check if it's correct.")

# File uploader to upload the image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Open and display the image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    
    # Preprocess the image for model prediction
    img = image.resize((224, 224))  # Resize to the model input size
    img = np.array(img)  # Convert the image to a numpy array
    img = img / 255.0  # Normalize the image
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    
    # Make prediction
    prediction = model.predict(img)
    
    # If prediction is greater than 0.5, it's correct, else incorrect
    if prediction[0] > 0.5:
        st.success("The uniform is correct! ✅")
    else:
        st.error("The uniform is incorrect! ❌")
