import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import os

# Load and preprocess the image
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(r"D:\Desktop\plant diease techsham\CNN_plantdiseases_model.keras")

def model_predict(image_path):
    model = load_model()
    img = cv2.imread(image_path)  # Read the file and convert it into an array
    H, W, C = 224, 224, 3
    img = cv2.resize(img, (H, W))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype("float32") / 255.0  # Normalize
    img = img.reshape(1, H, W, C)  # Reshape for the model

    prediction = np.argmax(model.predict(img), axis=-1)[0]
    return prediction

# Sidebar
st.sidebar.title("🌱 Plant Disease Detection")
st.sidebar.markdown(
    """
    <div style="color: #2C6B2F; font-size: 18px; font-family: Arial, sans-serif;">
        **Creator:** *Barsha Rani Sahoo*  
        This app helps in identifying plant diseases to support sustainable agriculture.
    </div>
    """, unsafe_allow_html=True
)
app_mode = st.sidebar.selectbox("Navigate", ["🏠 Home", "🔍 Disease Detection"])

# Class names
class_names = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
    'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew',
    'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
    'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy',
    'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
    'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot',
    'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy',
    'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy',
    'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew',
    'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot',
    'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
    'Tomato___healthy'
]

# Home Page
if app_mode == "🏠 Home":
    st.markdown(
        """
        <div style="text-align: center; font-family: Arial, sans-serif; color: #2C6B2F;">
            <h1>🌾 Plant Disease Detection System 🌾</h1>
            <p style="font-size: 20px;">
                Welcome to the Plant Disease Detection System!  
                This app leverages deep learning to identify plant diseases from leaf images, supporting sustainable agriculture practices.
            </p>
            <hr style="border-top: 3px solid #2C6B2F;">
            <p style="font-size: 18px;">Developed by: <b>Barsha Rani Sahoo</b></p>
            <div style="margin-top: 20px; font-size: 18px; font-weight: 500;">How it works:</div>
            <p style="font-size: 16px;">
                1. Upload an image of a plant leaf. <br>
                2. The model analyzes the leaf for diseases. <br>
                3. View the result and take necessary actions for the plant.
            </p>
        </div>
        """, unsafe_allow_html=True
    )
    st.image(
        "D:\Desktop\plant diease techsham\ghghv.jpg",
        caption="Healthy and Diseased Plant Leaves",
        use_container_width=True,  # Fixed for deprecation
    )

# Disease Detection Page
elif app_mode == "🔍 Disease Detection":
    st.header("🔍 Upload an Image to Detect Disease")
    st.markdown(
        """
        <p style="font-size: 16px; color: #2C6B2F; font-family: Arial, sans-serif;">
        Upload an image of a plant leaf, and the model will analyze and detect the disease (if any).  
        Ensure the image is clear and shows the leaf prominently.
        </p>
        """, unsafe_allow_html=True
    )

    test_image = st.file_uploader("Choose an Image:", type=["jpg", "jpeg", "png"])

    if test_image is not None:
        # Save the uploaded file temporarily
        save_path = os.path.join(os.getcwd(), test_image.name)
        with open(save_path, "wb") as f:
            f.write(test_image.getbuffer())

        # Display uploaded image
        st.image(test_image, use_container_width=True, caption="Uploaded Image")  # Fixed deprecation issue

        # Predict button
        if st.button("🚀 Predict"):
            st.info("Analyzing the image... Please wait.")
            result_index = model_predict(save_path)
            prediction = class_names[result_index]

            st.success(f"🌟 Prediction: The plant is **{prediction}**.")
            st.balloons()

            # Display additional info
            st.markdown(
                f"""
                <div style="text-align: center; font-size: 16px; color: #2C6B2F; font-family: Arial, sans-serif; margin-top: 20px;">
                    You can take necessary actions based on the diagnosis. 
                    Learn more about managing this disease at:
                </div>
                <div style="text-align: center;">
                    <a href="https://www.gardeningknowhow.com/" target="_blank" style="font-size: 18px; color: #2C6B2F;">Gardening Know How</a>
                </div>
                """, unsafe_allow_html=True
            )

# Footer
st.markdown(
    """
    <hr style="border-top: 3px solid #2C6B2F;">
    <p style="text-align: center; font-size: 14px; font-style: italic; color: #2C6B2F;">
    🌱 <i>Plant Disease Detection System</i> | Developed by <b>Barsha Rani Sahoo</b> © 2025  
    Powered by TensorFlow and Streamlit
    </p>
    """, unsafe_allow_html=True
)
