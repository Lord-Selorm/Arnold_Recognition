"""
WORKING ARNOLD RECOGNITION APP - Python 3.13 Compatible
Uses OpenCV for face detection, DeepFace for recognition
"""

import streamlit as st
import numpy as np
from pathlib import Path
import cv2
from deepface import DeepFace
from PIL import Image
import time
import os

# CSS
st.markdown("""
<style>
.stApp {
    background: #f8f9fa;
}
.main-header {
    text-align: center;
    color: #2c3e50;
    margin-bottom: 2rem;
    padding: 2rem;
    border-radius: 10px;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
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
    padding: 1rem;
    margin: 0.5rem 0;
    border-radius: 5px;
}
.not-arnold-result {
    background: linear-gradient(135deg, #f8d7da 0%, #f5c6cb 100%);
    border-left: 5px solid #dc3545;
    padding: 1rem;
    margin: 0.5rem 0;
    border-radius: 5px;
}
.upload-box {
    background: white;
    padding: 2rem;
    border-radius: 10px;
    text-align: center;
    border: 2px dashed #6c757d;
    margin: 2rem 0;
}
.stButton > button {
    background: linear-gradient(135deg, #007bff 0%, #0056b3 100%);
    color: white;
    border: none;
    padding: 0.75rem 2rem;
    border-radius: 8px;
    font-weight: 600;
}
</style>
""", unsafe_allow_html=True)

# Page config
st.set_page_config(
    page_title="Arnold Facial Recognition",
    page_icon="🏋️",
    layout="wide"
)

# Header
st.markdown("""
<div class="main-header">
    <h1>🏋️ Arnold Facial Recognition</h1>
    <p>Advanced Computer Vision & Machine Learning</p>
</div>
""", unsafe_allow_html=True)

def load_real_arnold_embeddings():
    """Load real Arnold embeddings using DeepFace"""
    try:
        arnold_embeddings = []
        arnold_path = Path("dataset/arnold")
        
        if not arnold_path.exists():
            st.error("❌ Arnold dataset not found!")
            return None
        
        # Get Arnold images
        arnold_images = list(arnold_path.glob("*.jpg")) + list(arnold_path.glob("*.png"))
        
        if len(arnold_images) == 0:
            st.error("❌ No Arnold images found in dataset!")
            return None
        
        # Process images for embeddings
        for img_path in arnold_images[:30]:  # Use first 30 for speed
            try:
                # Load image
                image = cv2.imread(str(img_path))
                if image is None:
                    continue
                    
                # Convert to RGB
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                # Extract embedding using DeepFace
                try:
                    embedding = DeepFace.represent(
                        image_rgb,
                        model_name='Facenet512',
                        enforce_detection=False
                    )[0]['embedding']
                    arnold_embeddings.append(np.array(embedding))
                    
                except:
                    continue
                
            except Exception as e:
                continue
        
        if len(arnold_embeddings) == 0:
            st.error("❌ Could not extract any Arnold embeddings!")
            return None
            
        return np.array(arnold_embeddings)
        
    except Exception as e:
        st.error(f"❌ Error loading Arnold data: {e}")
        return None

def detect_and_extract_faces(image):
    """Detect faces using OpenCV and extract embeddings"""
    try:
        # Convert to RGB
        if isinstance(image, Image.Image):
            image_array = np.array(image.convert('RGB'))
        else:
            image_array = image
            
        # Load face cascade
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Convert to grayscale for detection
        gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
        
        # Detect faces
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        
        if len(faces) == 0:
            return None, "No faces detected in the image"
        
        face_results = []
        
        # Process each face
        for (x, y, w, h) in faces:
            # Add padding
            padding = 20
            x1 = max(0, x - padding)
            y1 = max(0, y - padding)
            x2 = min(image_array.shape[1], x + w + padding)
            y2 = min(image_array.shape[0], y + h + padding)
            
            # Crop face
            face_crop = image_array[y1:y2, x1:x2]
            
            # Extract embedding using DeepFace
            try:
                embedding = DeepFace.represent(
                    face_crop,
                    model_name='Facenet512',
                    enforce_detection=False
                )[0]['embedding']
                
                face_results.append({
                    'box': (x, y, w, h),
                    'embedding': np.array(embedding)
                })
            except:
                continue
        
        return face_results, None
        
    except Exception as e:
        return None, f"Error processing image: {e}"

def is_real_arnold(face_embedding, arnold_embeddings):
    """Check if face is Arnold using similarity"""
    if arnold_embeddings is None:
        return False, 0.0, "Arnold data not loaded"
    
    try:
        # Calculate similarities
        similarities = []
        for arnold_embedding in arnold_embeddings:
            similarity = np.dot(face_embedding, arnold_embedding) / (
                np.linalg.norm(face_embedding) * np.linalg.norm(arnold_embedding)
            )
            similarities.append(similarity)
        
        # Get best match
        max_sim = max(similarities)
        avg_sim = np.mean(similarities)
        
        # Threshold logic
        threshold = 0.28
        if max_sim > 0.4:
            threshold = 0.35
        elif avg_sim > 0.3:
            threshold = 0.32
        
        # Final decision
        is_arnold = max_sim >= threshold
        confidence = min(max_sim + 0.1, 0.95)
        
        # Analysis
        if is_arnold:
            if max_sim > 0.8:
                analysis = "Excellent match - Strong Arnold features detected"
            elif max_sim > 0.6:
                analysis = "Good match - Clear Arnold characteristics"
            else:
                analysis = "Moderate match - Some Arnold features present"
        else:
            analysis = "Not Arnold - Different facial characteristics"
        
        return is_arnold, confidence, analysis
        
    except Exception as e:
        return False, 0.0, f"Error in comparison: {e}"

def draw_results_on_image(image_array, face_results, arnold_embeddings):
    """Draw detection boxes and labels on image"""
    result_image = image_array.copy()
    
    for i, face_result in enumerate(face_results):
        x, y, w, h = face_result['box']
        
        # Check if Arnold
        is_arnold, confidence, analysis = is_real_arnold(
            face_result['embedding'], 
            arnold_embeddings
        )
        
        # Draw rectangle
        if is_arnold:
            color = (0, 255, 0)  # Green for Arnold
            label = f"Arnold {confidence:.2f}"
        else:
            color = (0, 0, 255)  # Red for not Arnold
            label = f"Not Arnold {confidence:.2f}"
        
        # Draw rectangle
        cv2.rectangle(result_image, (x, y), (x+w, y+h), color, 2)
        
        # Draw label
        cv2.putText(result_image, label, (x, y-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    
    return result_image

# Load Arnold data
if 'arnold_embeddings' not in st.session_state:
    with st.spinner("Loading Arnold reference data..."):
        st.session_state.arnold_embeddings = load_real_arnold_embeddings()

# Main content
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown('<div class="upload-box">', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("📸 Choose an image...", type=['jpg', 'jpeg', 'png'])
    st.markdown('</div>', unsafe_allow_html=True)
    
    if uploaded_file is not None:
        # Display uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        
        # Analyze button
        if st.button("🔍 Analyze Image", type="primary"):
            with st.spinner("Analyzing faces..."):
                # Detect and extract faces
                face_results, error = detect_and_extract_faces(image)
                
                if error:
                    st.error(f"❌ {error}")
                elif face_results:
                    st.success(f"✅ Detected {len(face_results)} face(s)")
                    
                    # Draw results on image
                    image_array = np.array(image.convert('RGB'))
                    result_image = draw_results_on_image(image_array, face_results, st.session_state.arnold_embeddings)
                    
                    # Display result image
                    st.image(result_image, caption="Face Detection Results", use_column_width=True)
                    
                    # Process each face
                    arnold_count = 0
                    total_faces = len(face_results)
                    
                    for i, face_result in enumerate(face_results):
                        # Check if Arnold
                        is_arnold, confidence, analysis = is_real_arnold(
                            face_result['embedding'], 
                            st.session_state.arnold_embeddings
                        )
                        
                        if is_arnold:
                            arnold_count += 1
                        
                        # Display result
                        if is_arnold:
                            st.markdown(f"""
                            <div class="result-box arnold-result">
                                <h4>🏋️ Face {i+1}: Arnold Detected</h4>
                                <p><strong>Confidence:</strong> {confidence:.2f}</p>
                                <p><strong>Analysis:</strong> {analysis}</p>
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.markdown(f"""
                            <div class="result-box not-arnold-result">
                                <h4>👤 Face {i+1}: Not Arnold</h4>
                                <p><strong>Confidence:</strong> {confidence:.2f}</p>
                                <p><strong>Analysis:</strong> {analysis}</p>
                            </div>
                            """, unsafe_allow_html=True)
                    
                    # Summary
                    st.markdown("---")
                    st.markdown("### 📊 Summary")
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Total Faces", total_faces)
                    with col2:
                        st.metric("Arnold Detected", arnold_count)
                    with col3:
                        st.metric("Not Arnold", total_faces - arnold_count)
                    
                else:
                    st.error("❌ No faces detected in image")

with col2:
    st.markdown("### 📋 System Status")
    
    if st.session_state.arnold_embeddings is not None:
        st.success(f"✅ {len(st.session_state.arnold_embeddings)} Arnold embeddings loaded")
    else:
        st.error("❌ Failed to load Arnold data")
    
    st.markdown("---")
    st.markdown("### 🎯 Features")
    st.markdown("""
    - **Multi-face detection**
    - **Real Arnold recognition**
    - **Visual result display**
    - **Confidence scores**
    - **Python 3.13 compatible**
    """)
    
    st.markdown("---")
    st.markdown("### 💡 Technology")
    st.markdown("""
    - **OpenCV** - Face detection
    - **DeepFace** - Feature extraction
    - **FaceNet512** - Embeddings
    - **Cosine similarity** - Matching
    """)

# Footer
st.markdown("---")
st.markdown('<p style="text-align: center; color: #6c757d;">🏋️ Arnold Facial Recognition System</p>', unsafe_allow_html=True)
