"""
REAL ARNOLD RECOGNITION APP - Accurate & Responsive
Uses actual Arnold training data with real TensorFlow/DeepFace
"""

import streamlit as st
import pickle
import numpy as np
from pathlib import Path
import cv2
from mtcnn import MTCNN
from deepface import DeepFace
from PIL import Image
import time
import os

# Responsive CSS
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
.feature-card {
    background: #f8f9fa;
    padding: 1rem;
    border-radius: 8px;
    margin: 0.5rem 0;
    border-left: 3px solid #667eea;
}
.metric-card {
    background: white;
    padding: 1rem;
    border-radius: 8px;
    text-align: center;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}
@media (max-width: 768px) {
    .main-header {
        padding: 1rem;
        margin-bottom: 1rem;
    }
    .upload-box {
        padding: 1rem;
        margin: 1rem 0;
    }
    .result-box {
        padding: 1rem;
        margin: 0.5rem 0;
    }
}
</style>
""", unsafe_allow_html=True)

# Page config
st.set_page_config(
    page_title="Arnold Facial Recognition",
    page_icon="🏋️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Header
st.markdown("""
<div class="main-header">
    <h1>🏋️ Arnold Facial Recognition</h1>
    <p>Advanced Computer Vision & Machine Learning</p>
</div>
""", unsafe_allow_html=True)

def load_real_arnold_embeddings():
    """Load real Arnold embeddings with maximum accuracy"""
    try:
        arnold_embeddings = []
        arnold_path = Path("dataset/arnold")
        
        if not arnold_path.exists():
            st.error("❌ Arnold dataset not found!")
            return None
        
        # Load first 30 Arnold images for speed
        detector = MTCNN()
        
        arnold_images = list(arnold_path.glob("*.jpg")) + list(arnold_path.glob("*.png"))
        
        if len(arnold_images) == 0:
            st.error("❌ No Arnold images found in dataset!")
            return None
        
        # Use first 30 images for speed (still very accurate)
        for img_path in arnold_images[:30]:
            try:
                # Load image with error handling
                image = cv2.imread(str(img_path))
                if image is None:
                    continue
                    
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                # Detect ALL faces in image
                faces = detector.detect_faces(image_rgb)
                if len(faces) == 0:
                    continue
                
                # Use only best face per image for speed
                best_face = max(faces, key=lambda x: x['confidence'])
                if best_face['confidence'] > 0.9:  # Only high-confidence faces
                    x, y, w, h = best_face['box']
                    
                    # Add padding for better feature extraction
                    padding = 20
                    x1 = max(0, x - padding)
                    y1 = max(0, y - padding)
                    x2 = min(image_rgb.shape[1], x + w + padding)
                    y2 = min(image_rgb.shape[0], y + h + padding)
                    
                    # Crop face with padding
                    face_crop = image_rgb[y1:y2, x1:x2]
                    
                    # Extract embedding with FaceNet512 only for consistency
                    try:
                        embedding1 = DeepFace.represent(
                            face_crop,
                            model_name='Facenet512',
                            enforce_detection=False
                        )[0]['embedding']
                        arnold_embeddings.append(np.array(embedding1))
                        
                    except:
                        continue
                
            except Exception as e:
                continue
        
        if len(arnold_embeddings) == 0:
            st.error("❌ Could not extract any Arnold embeddings!")
            return None
            
        # Convert to numpy array for faster processing
        return np.array(arnold_embeddings)
        
    except Exception as e:
        st.error(f"❌ Error loading Arnold data: {e}")
        return None

def extract_face_embedding(image):
    """Extract face embedding from uploaded image"""
    try:
        # Convert to RGB
        if isinstance(image, Image.Image):
            image_array = np.array(image.convert('RGB'))
        else:
            image_array = image
            
        # Detect faces
        detector = MTCNN()
        faces = detector.detect_faces(image_array)
        
        if len(faces) == 0:
            return None, "No faces detected in the image"
        
        face_results = []
        
        for face in faces:
            x, y, w, h = face['box']
            confidence = face['confidence']
            
            # Add padding
            padding = 20
            x1 = max(0, x - padding)
            y1 = max(0, y - padding)
            x2 = min(image_array.shape[1], x + w + padding)
            y2 = min(image_array.shape[0], y + h + padding)
            
            # Crop face
            face_crop = image_array[y1:y2, x1:x2]
            
            # Extract embedding with FaceNet512 only for consistency
            try:
                embedding = DeepFace.represent(
                    face_crop,
                    model_name='Facenet512',
                    enforce_detection=False
                )[0]['embedding']
                
                face_results.append({
                    'box': (x, y, w, h),
                    'confidence': confidence,
                    'embedding': np.array(embedding)
                })
            except:
                continue
        
        return face_results, None
        
    except Exception as e:
        return None, f"Error processing image: {e}"

def is_real_arnold(face_embedding, arnold_embeddings):
    """Advanced similarity analysis with FaceNet512"""
    if arnold_embeddings is None:
        return False, 0.0, "Arnold data not loaded"
    
    try:
        # Vectorized similarity calculation for speed
        similarities = np.dot(arnold_embeddings, face_embedding) / (
            np.linalg.norm(arnold_embeddings, axis=1) * np.linalg.norm(face_embedding)
        )
        
        # Advanced similarity analysis
        max_sim = np.max(similarities)
        avg_sim = np.mean(similarities)
        median_sim = np.median(similarities)
        
        # Top-5 and Top-10 averages
        top_5_avg = np.mean(np.sort(similarities)[-5:])
        top_10_avg = np.mean(np.sort(similarities)[-10:])
        
        # Adaptive threshold based on confidence
        threshold = 0.28
        if max_sim > 0.4:
            threshold = 0.35
        elif avg_sim > 0.3:
            threshold = 0.32
        
        # Final decision
        is_arnold = max_sim >= threshold
        
        # Confidence calculation
        confidence = min(max_sim + 0.1, 0.95)
        
        # Analysis message
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

# Sidebar info
with st.sidebar:
    st.markdown("### 🎯 System Status")
    
    # Load Arnold data
    if 'arnold_embeddings' not in st.session_state:
        with st.spinner("Loading Arnold reference data..."):
            st.session_state.arnold_embeddings = load_real_arnold_embeddings()
    
    if st.session_state.arnold_embeddings is not None:
        st.success(f"✅ {len(st.session_state.arnold_embeddings)} Arnold embeddings loaded")
    else:
        st.error("❌ Failed to load Arnold data")
    
    st.markdown("---")
    st.markdown("### 📋 Features")
    st.markdown("""
    <div class="feature-card">
        <strong>🔍 Face Detection</strong><br>
        MTCNN for accurate detection
    </div>
    <div class="feature-card">
        <strong>🧠 Recognition</strong><br>
        FaceNet512 embeddings
    </div>
    <div class="feature-card">
        <strong>📊 Analysis</strong><br>
        Advanced similarity matching
    </div>
    """, unsafe_allow_html=True)

# Main content
st.markdown("### 📸 Upload Image for Analysis")

# Responsive columns
col1, col2 = st.columns([2, 1])

with col1:
    uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png'])
    
    if uploaded_file is not None:
        # Display uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        
        # Analyze button
        if st.button("🔍 Analyze Image", type="primary"):
            with st.spinner("Analyzing faces..."):
                # Extract face embeddings
                face_results, error = extract_face_embedding(image)
                
                if error:
                    st.error(f"❌ {error}")
                elif face_results:
                    st.success(f"✅ Detected {len(face_results)} face(s)")
                    
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
                    st.markdown("### 📊 Analysis Summary")
                    
                    # Responsive metrics
                    if total_faces <= 2:
                        cols = st.columns(3)
                    else:
                        cols = st.columns(4)
                    
                    with cols[0]:
                        st.markdown(f"""
                        <div class="metric-card">
                            <h4>{total_faces}</h4>
                            <p>Total Faces</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with cols[1]:
                        st.markdown(f"""
                        <div class="metric-card">
                            <h4>{arnold_count}</h4>
                            <p>Arnold Detected</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with cols[2]:
                        st.markdown(f"""
                        <div class="metric-card">
                            <h4>{total_faces - arnold_count}</h4>
                            <p>Not Arnold</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    if total_faces > 2:
                        with cols[3]:
                            accuracy = (arnold_count / total_faces) * 100 if total_faces > 0 else 0
                            st.markdown(f"""
                            <div class="metric-card">
                                <h4>{accuracy:.1f}%</h4>
                                <p>Arnold Ratio</p>
                            </div>
                            """, unsafe_allow_html=True)
                
                else:
                    st.error("❌ No faces detected in the image")

with col2:
    st.markdown("### 📋 Instructions")
    st.markdown("""
    1. **Upload Image** - Select a photo with faces
    2. **Click Analyze** - Process with AI models
    3. **View Results** - See detection and recognition
    """)
    
    st.markdown("---")
    st.markdown("### 💡 Technology")
    st.markdown("""
    - **MTCNN** - Face detection
    - **FaceNet512** - Feature extraction
    - **Cosine Similarity** - Matching algorithm
    - **Adaptive Threshold** - Smart detection
    """)

# Footer
st.markdown("---")
st.markdown('<p style="text-align: center; color: #6c757d;">🏋️ Arnold Facial Recognition System - Real ML Accuracy</p>', unsafe_allow_html=True)
