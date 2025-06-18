# streamlit_mnist_app.py
import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt

# Load the trained model
@st.cache_resource
def load_model():
    """Load the pre-trained MNIST model"""
    try:
        model = tf.keras.models.load_model('mnist_cnn_model.h5')
        return model
    except:
        st.error("Model file not found. Please train and save the model first.")
        return None

def preprocess_image(image):
    """Preprocess uploaded image for prediction"""
    # Convert to grayscale
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # Resize to 28x28
    image = cv2.resize(image, (28, 28))
    
    # Invert if needed (MNIST digits are white on black)
    if np.mean(image) > 127:
        image = 255 - image
    
    # Normalize
    image = image.astype('float32') / 255.0
    
    # Reshape for model
    image = image.reshape(1, 28, 28, 1)
    
    return image

def main():
    st.title("🔢 MNIST Digit Classifier")
    st.write("Upload a handwritten digit image or draw one to get predictions!")
    
    # Load model
    model = load_model()
    if model is None:
        return
    
    # Sidebar
    st.sidebar.header("Options")
    input_method = st.sidebar.selectbox(
        "Choose input method:",
        ["Upload Image", "Draw Digit", "Use Sample"]
    )
    
    if input_method == "Upload Image":
        uploaded_file = st.file_uploader(
            "Choose an image...", 
            type=['png', 'jpg', 'jpeg']
        )
        
        if uploaded_file is not None:
            # Display uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption='Uploaded Image', width=200)
            
            # Preprocess and predict
            processed_image = preprocess_image(np.array(image))
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Processed Image")
                st.image(processed_image.reshape(28, 28), width=200, clamp=True)
            
            with col2:
                st.subheader("Prediction")
                prediction = model.predict(processed_image)
                predicted_digit = np.argmax(prediction)
                confidence = prediction[0][predicted_digit]
                
                st.metric("Predicted Digit", predicted_digit)
                st.metric("Confidence", f"{confidence:.2%}")
                
                # Show all probabilities
                st.subheader("All Probabilities")
                prob_df = {
                    'Digit': list(range(10)),
                    'Probability': prediction[0]
                }
                st.bar_chart(prob_df['Probability'])
    
    elif input_method == "Draw Digit":
        st.info("Drawing functionality would require additional JavaScript components. For demo purposes, use the upload option.")
    
    else:  # Use Sample
        st.subheader("Sample MNIST Images")
        
        # Load sample images (you'd load these from your test set)
        sample_images = generate_sample_images()  # This would be your actual test images
        
        selected_sample = st.selectbox("Choose a sample:", range(len(sample_images)))
        
        if st.button("Predict Sample"):
            sample_image = sample_images[selected_sample]
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.image(sample_image.reshape(28, 28), width=200, clamp=True)
            
            with col2:
                prediction = model.predict(sample_image.reshape(1, 28, 28, 1))
                predicted_digit = np.argmax(prediction)
                confidence = prediction[0][predicted_digit]
                
                st.metric("Predicted Digit", predicted_digit)
                st.metric("Confidence", f"{confidence:.2%}")

def generate_sample_images():
    """Generate sample images for demo"""
    # In real implementation, load from your test set
    return [np.random.rand(28, 28) for _ in range(5)]

if __name__ == "__main__":
    main()