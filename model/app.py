import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications import mobilenet_v2
import numpy as np
from PIL import Image
import cv2
from streamlit_webrtc import webrtc_streamer, RTCConfiguration, VideoFrameCallback
import av

# --- Constants ---
IMG_WIDTH, IMG_HEIGHT = 224, 224
MODEL_PATH = 'model.h5'  # Make sure 'model.h5' is in the same directory

# --- Model Loading ---
@st.cache_resource
def load_keras_model():
    """Loads the pre-trained Keras model from disk."""
    try:
        model = load_model(MODEL_PATH)
        return model
    except Exception as e:
        st.error(f"Error loading model from {MODEL_PATH}: {e}")
        st.error("Please make sure the 'model.h5' file is in the same directory as this app.")
        return None

# --- Face Detector Loading ---
@st.cache_resource
def load_face_cascade():
    """Loads the OpenCV Haar Cascade for face detection."""
    try:
        # Use the built-in cascade file from OpenCV
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        if not cv2.os.path.exists(cascade_path):
            st.error("Could not find Haar cascade file. Please ensure OpenCV is correctly installed.")
            return None
        face_cascade = cv2.CascadeClassifier(cascade_path)
        return face_cascade
    except Exception as e:
        st.error(f"Error loading face cascade classifier: {e}")
        return None

# --- Image Preprocessing (Unchanged from your original) ---
def preprocess_image(image_pil):
    """
    Preprocesses a PIL image to be ready for the MobileNetV2 model.
    """
    try:
        # Resize to model's expected input size
        image_resized = image_pil.resize((IMG_WIDTH, IMG_HEIGHT))
        
        # Convert to numpy array
        image_array = np.array(image_resized)
        
        # Ensure it's 3 channels (in case of grayscale)
        if image_array.ndim == 2:
            image_array = cv2.cvtColor(image_array, cv2.COLOR_GRAY2RGB)
            
        # Ensure it's RGB (in case of RGBA)
        if image_array.shape[2] == 4:
            image_array = cv2.cvtColor(image_array, cv2.COLOR_RGBA2RGB)

        # Apply the specific MobileNetV2 preprocessing
        preprocessed_image = mobilenet_v2.preprocess_input(image_array)
        
        # Expand dimensions to create a batch of 1
        batch_image = np.expand_dims(preprocessed_image, axis=0)
        
        return batch_image
    except Exception as e:
        st.error(f"Error preprocessing image: {e}")
        return None

# --- WebRTC Video Processing Class ---
class VideoProcessor:
    def __init__(self, model, face_cascade):
        self.model = model
        self.face_cascade = face_cascade
        self.threshold = 0.5  # As seen in your notebook

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        try:
            # Convert frame to numpy array (BGR format)
            img = frame.to_ndarray(format="bgr24")
            
            # Convert to grayscale for face detection
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Detect faces
            faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            
            for (x, y, w, h) in faces:
                # Crop the face (with a little padding)
                face_crop_bgr = img[y:y+h, x:x+w]
                
                if face_crop_bgr.size == 0:
                    continue

                # --- Prediction ---
                # 1. Convert to PIL (model expects PIL)
                #    Must convert from BGR (cv2) to RGB (PIL)
                face_crop_rgb = cv2.cvtColor(face_crop_bgr, cv2.COLOR_BGR2RGB)
                face_pil = Image.fromarray(face_crop_rgb)
                
                # 2. Preprocess the PIL image
                batch_image = preprocess_image(face_pil)
                
                if batch_image is not None:
                    # 3. Predict
                    prediction = self.model.predict(batch_image)
                    prediction_value = prediction[0][0]
                    
                    # 4. Get label and color
                    if prediction_value > self.threshold:
                        label = f"Spoof ({prediction_value:.2f})"
                        color = (0, 0, 255)  # Red for Spoof
                    else:
                        label = f"Real ({prediction_value:.2f})"
                        color = (0, 255, 0)  # Green for Real
                        
                    # 5. Draw bounding box and label on the *original* frame (img)
                    cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)
                    cv2.putText(img, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            # Convert annotated frame back to VideoFrame
            return av.VideoFrame.from_ndarray(img, format="bgr24")
        
        except Exception as e:
            st.error(f"Error in video processing: {e}")
            # Return original frame on error
            return av.VideoFrame.from_ndarray(frame.to_ndarray(format="bgr24"), format="bgr24")

# --- STUN server configuration (for deployment) ---
RTC_CONFIG = RTCConfiguration({
    "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
})

# --- Main Streamlit App ---
def main():
    st.title("Face Anti-Spoofing Detector 🧑‍-")

    # Load models
    model = load_keras_model()
    face_cascade = load_face_cascade()
    
    if model is None or face_cascade is None:
        st.warning("Model or Face Cascade could not be loaded. App cannot proceed.")
        st.stop()

    # --- Sidebar for Mode Selection ---
    st.sidebar.title("Select Mode")
    app_mode = st.sidebar.radio(
        "Choose the app mode",
        ["Upload Image", "Live Webcam Detection"]
    )

    # --- Mode 1: Upload Image ---
    if app_mode == "Upload Image":
        st.info("Upload an image to classify it as **Real** or **Spoof**.")
        
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
        
        if uploaded_file is not None:
            try:
                # Open and display the uploaded image
                image_pil = Image.open(uploaded_file).convert('RGB')
                st.image(image_pil, caption='Uploaded Image', use_column_width=True)
            
            except Exception as e:
                st.error(f"Error opening image file: {e}")
                st.stop()

            # Preprocess the image and make prediction
            with st.spinner('Analyzing...'):
                batch_image = preprocess_image(image_pil)
                
                if batch_image is not None:
                    # Make prediction
                    prediction = model.predict(batch_image)
                    prediction_value = prediction[0][0]
                    
                    threshold = 0.5 
                    
                    if prediction_value > threshold:
                        label = "Spoof"
                        st.error(f"**Prediction: {label}**")
                    else:
                        label = "Real"
                        st.success(f"**Prediction: {label}**")
                    
                    st.write(f"Model Score: `{prediction_value:.4f}`")
                    st.caption(f"(Score > {threshold} is classified as Spoof)")

    # --- Mode 2: Live Webcam Detection ---
    elif app_mode == "Live Webcam Detection":
        st.info("Please allow webcam access. The app will detect faces and classify them in real-time.")
        
        webrtc_streamer(
            key="anti-spoofing-detector",
            video_frame_callback=VideoProcessor(model=model, face_cascade=face_cascade).recv,
            rtc_configuration=RTC_CONFIG,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True
        )

if __name__ == "__main__":
    main()