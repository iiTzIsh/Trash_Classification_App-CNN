import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
from PIL import Image

# Load model
model = load_model('trashnet_model.h5')

# Class labels
class_labels = ['Cardboard', 'Glass', 'Metal', 'Paper', 'Plastic', 'Trash']

# Disposal instructions
disposal_instructions = {
    'Cardboard': '🟤 **Cardboard:** Flatten cardboard boxes and keep them dry before placing them in the recycling bin. Avoid contamination with food or liquids.',
    'Glass': '🔵 **Glass:** Rinse glass bottles and containers. Separate by color if required. Do not mix broken glass with general waste.',
    'Metal': '⚙️ **Metal:** Rinse cans and metal containers. Remove sharp edges and dispose of responsibly in a metal recycling bin.',
    'Paper': '📄 **Paper:** Recycle newspapers, magazines, and clean paper products. Avoid recycling wet or greasy paper.',
    'Plastic': '🟢 **Plastic:** Rinse plastic bottles and containers. Follow local recycling codes for specific types (e.g., PET, HDPE).',
    'Trash': '🗑️ **Trash:** Dispose of non-recyclable waste responsibly. Consider reducing waste by reusing items or composting biodegradable waste.'
}

# Custom styles
st.set_page_config(page_title="Trash Classifier", page_icon="♻️", layout="wide")
st.markdown("""
    <style>
    .main {
        background-color: #f5f5f5;
    }
    .stButton > button {
        background-color: #4CAF50;
        color: white;
        border-radius: 8px;
        padding: 10px 20px;
        font-size: 16px;
    }
    .stButton > button:hover {
        background-color: #45a049;
    }
    </style>
    """, unsafe_allow_html=True)

# App title and description
st.title("♻️ Trash Classification App")
st.markdown("""
Welcome to the **Trash Classification App**! Upload an image of waste, and we'll classify its type and provide **disposal instructions** to help you manage waste responsibly.

---

""")

# File uploader
st.header("Step 1: Upload an Image")
uploaded_file = st.file_uploader("Choose a file", type=["jpg", "jpeg", "png"])

# Sidebar content
st.sidebar.header("About This App")
st.sidebar.info("""
This app uses a **Convolutional Neural Network (CNN)** trained on the **TrashNet Dataset** to classify waste into categories:

🌟 Features:
- Upload an image of trash .
- Get the **category** and **confidence level** of the prediction.
- Learn **how to dispose** of the trash properly.
- Explore **eco-friendly tips** to reduce waste.

""")

# Sidebar eco tip
st.sidebar.subheader("🌱 Eco Tip")
st.sidebar.write("Reuse items whenever possible. Small actions lead to big changes!")

if uploaded_file is not None:
    # Preprocess the uploaded image
    image = Image.open(uploaded_file)

    # Resize the image for display
    resized_image = image.resize((300, 300))  # Resize for display purposes
    st.image(resized_image, caption="Uploaded Image", use_container_width=False, width=300)

    
    # Convert image to required format
    image = image.resize((224, 224))  # Resize for the model
    image_array = img_to_array(image)  # Convert image to array
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
    image_array = image_array / 255.0  # Normalize pixel values

    # Predict the category
    prediction = model.predict(image_array)
    predicted_class = class_labels[np.argmax(prediction)]
    confidence = np.max(prediction)

    # Display prediction
    st.header("Step 2: Results")
    st.success(f"**Prediction:** {predicted_class}")
    st.write(f"**Confidence:** {confidence * 100:.2f}%")

    # Display disposal instructions
    st.header("Step 3: How to Dispose")
    st.write(disposal_instructions[predicted_class])
    
else:
    st.warning("Please upload an image to proceed.")

# Footer
st.markdown("""
---
👩‍💻 **Developed by:** Ishara Madusanka 
""")
