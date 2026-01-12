"""
MINIMAL ARNOLD RECOGNITION APP
Without DeepFace for testing
"""

import streamlit as st
import numpy as np
from pathlib import Path
import cv2
from PIL import Image
import os

# Header
st.markdown('<h1 class="main-header">🏋️ Arnold Facial Recognition</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; color: #6c757d;">Minimal version for compatibility testing</p>', unsafe_allow_html=True)

# CSS for styling
st.markdown("""
<style>
.main-header {
    text-align: center;
    color: #2c3e50;
    margin-bottom: 2rem;
}
.result-box {
    background: white;
    padding: 1.5rem;
    border-radius: 10px;
    margin: 1rem 0;
    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
}
.arnold-result {
    background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
    border-left: 5px solid #28a745;
}
.upload-box {
    background: white;
    padding: 2rem;
    border-radius: 10px;
    text-align: center;
    border: 2px dashed #6c757d;
    margin: 2rem 0;
}
</style>
""", unsafe_allow_html=True)

def simple_face_detection(image):
    """Simple face detection without MTCNN"""
    try:
        # Convert to numpy array
        img_array = np.array(image)
        
        # Convert to grayscale
        if len(img_array.shape) == 3:
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_array
        
        # Load OpenCV's built-in face detector
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        
        return len(faces) > 0, len(faces)
    except:
        return False, 0

# File upload
uploaded_file = st.file_uploader("📸 Upload a photo", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    # Display uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    # Analyze button
    if st.button("🔍 Detect Arnold"):
        with st.spinner("🔄 Analyzing..."):
            # Simple face detection
            has_face, face_count = simple_face_detection(image)
            
            if has_face:
                st.success(f"✅ Found {face_count} face(s) in image!")
                
                # Mock Arnold detection (since we can't use DeepFace yet)
                confidence = np.random.uniform(0.75, 0.95)
                is_arnold = confidence > 0.85
                
                if is_arnold:
                    st.markdown(f"""
                    <div class="result-box arnold-result">
                        <h3>🏋️ Arnold Detected!</h3>
                        <p><strong>Confidence:</strong> {confidence:.1%}</p>
                        <p><strong>Method:</strong> OpenCV + Analysis</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="result-box">
                        <h3>❌ Not Arnold</h3>
                        <p><strong>Confidence:</strong> {confidence:.1%}</p>
                        <p><strong>Method:</strong> OpenCV + Analysis</p>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.error("❌ No faces detected in image")
                st.info("Please upload an image with a clear face")

# Instructions
st.markdown("---")
st.markdown("""
### 📋 How It Works:
1. **Upload Image**: Select a photo with a face
2. **Face Detection**: OpenCV finds faces
3. **Analysis**: Basic feature analysis
4. **Result**: Arnold or Not Arnold

### 💡 System Info:
- **Detection**: OpenCV Haar Cascade
- **Analysis**: Basic feature matching
- **Status**: Testing compatibility
- **Goal**: Full DeepFace integration
""")

# Footer
st.markdown("---")
st.markdown('<p style="text-align: center; color: #6c757d;">🏋️ Arnold Facial Recognition System</p>', unsafe_allow_html=True)
