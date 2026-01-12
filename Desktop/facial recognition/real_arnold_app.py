"""
REAL ARNOLD RECOGNITION APP
Uses actual Arnold training data instead of synthetic embeddings
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

# Clean CSS
st.markdown("""
<style>
.stApp {
    background: #f8f9fa;
}
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
.not-arnold-result {
    background: linear-gradient(135deg, #f8d7da 0%, #f5c6cb 100%);
    border-left: 5px solid #dc3545;
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
.stButton > button:hover {
    background: linear-gradient(135deg, #0056b3 0%, #004085 100%);
}
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<h1 class="main-header">🏋️ Arnold Facial Recognition</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; color: #6c757d;">Optimized for speed with high accuracy</p>', unsafe_allow_html=True)

def load_real_arnold_embeddings():
    """Load real Arnold embeddings with maximum accuracy optimizations"""
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
        for img_path in arnold_images[:30]:  # Optimized for speed
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

def extract_face_embedding(image_array):
    """Extract face embeddings with maximum accuracy optimizations"""
    try:
        detector = MTCNN()
        
        # Detect all faces
        faces = detector.detect_faces(image_array)
        if len(faces) == 0:
            return [], "No face detected"
        
        face_results = []
        
        for i, face in enumerate(faces):
            # Only process high-confidence faces
            if face['confidence'] < 0.85:
                continue
                
            x, y, w, h = face['box']
            confidence = face['confidence']
            
            # Add padding for better feature extraction
            padding = 20
            x1 = max(0, x - padding)
            y1 = max(0, y - padding)
            x2 = min(image_array.shape[1], x + w + padding)
            y2 = min(image_array.shape[0], y + h + padding)
            
            # Crop face with padding
            face_crop = image_array[y1:y2, x1:x2]
            
            # Extract embeddings with FaceNet512 only for consistency
            embeddings = []
            
            # FaceNet512 (most accurate and consistent)
            try:
                embedding1 = DeepFace.represent(
                    face_crop,
                    model_name='Facenet512',
                    enforce_detection=False
                )[0]['embedding']
                embeddings.append(np.array(embedding1))
            except:
                pass
            
            # Only add face if we got at least one good embedding
            if embeddings:
                # Use the best embedding (highest quality)
                best_embedding = embeddings[0]  # FaceNet512 is usually best
                
                face_results.append({
                    'face_id': i + 1,
                    'embedding': best_embedding,
                    'confidence': confidence,
                    'box': (x, y, w, h)
                })
        
        return face_results, f"Found {len(face_results)} high-confidence face(s)"
        
    except Exception as e:
        return [], f"Error: {str(e)}"

def is_real_arnold(embedding, arnold_embeddings):
    """ULTIMATE ACCURACY: Advanced similarity analysis with FaceNet512"""
    try:
        if arnold_embeddings is None:
            return False, 0.0, "No Arnold data"
        
        # Calculate similarities to Arnold embeddings (optimized for speed)
        # Use vectorized operations for speed
        embedding_norm = np.linalg.norm(embedding)
        arnold_norms = np.linalg.norm(arnold_embeddings, axis=1)
        
        # Vectorized cosine similarity calculation
        similarities = np.dot(arnold_embeddings, embedding) / (arnold_norms * embedding_norm)
        
        # Advanced similarity analysis
        max_similarity = max(similarities)
        avg_similarity = np.mean(similarities)
        median_similarity = np.median(similarities)
        
        # Multiple similarity metrics for robustness
        top5_avg = np.mean(sorted(similarities, reverse=True)[:5])
        top10_avg = np.mean(sorted(similarities, reverse=True)[:10])
        
        # Adaptive threshold based on similarity distribution
        if max_similarity > 0.5:  # Very high confidence
            threshold = 0.28
        elif max_similarity > 0.4:  # High confidence
            threshold = 0.30
        elif max_similarity > 0.3:  # Medium confidence
            threshold = 0.32
        else:  # Low confidence
            threshold = 0.35
        
        # Additional quality checks
        consistency_score = avg_similarity / max_similarity if max_similarity > 0 else 0
        quality_factor = min(1.0, len(similarities) / 50.0)  # More references = higher confidence
        
        # Final decision with multiple factors
        final_score = (max_similarity * 0.6 + 
                     avg_similarity * 0.2 + 
                     median_similarity * 0.1 + 
                     consistency_score * 0.1) * quality_factor
        
        is_arnold = final_score > threshold
        
        # Detailed analysis message
        if is_arnold:
            if max_similarity > 0.6:
                analysis = f"Excellent match! ({max_similarity:.1%} max, {avg_similarity:.1%} avg)"
            elif max_similarity > 0.4:
                analysis = f"Strong match ({max_similarity:.1%} max, {avg_similarity:.1%} avg)"
            else:
                analysis = f"Likely match ({max_similarity:.1%} max, {avg_similarity:.1%} avg)"
        else:
            if max_similarity > 0.25:
                analysis = f"Possible but low confidence ({max_similarity:.1%} max, {avg_similarity:.1%} avg)"
            else:
                analysis = f"Low similarity ({max_similarity:.1%} max, {avg_similarity:.1%} avg)"
        
        return is_arnold, final_score, analysis
            
    except Exception as e:
        return False, 0.0, f"Error: {str(e)}"

# Load real Arnold embeddings
arnold_embeddings = load_real_arnold_embeddings()

# File upload
uploaded_file = st.file_uploader("📸 Upload a photo", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    # Display uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    # Analyze button
    if st.button("🔍 Detect Arnold"):
        if arnold_embeddings is None:
            st.error("❌ Cannot analyze - no Arnold reference data available!")
        else:
            with st.spinner("🔄 Analyzing..."):
                # Convert to RGB
                image_rgb = np.array(image.convert('RGB'))
                
                # Extract embeddings for all faces
                face_results, message = extract_face_embedding(image_rgb)
                
                if len(face_results) > 0:
                    # Process each detected face
                    st.markdown("---")
                    st.markdown("### 🎯 Analysis Results")
                    st.info(f"📸 {message}")
                    
                    for i, face_data in enumerate(face_results):
                        # Check against real Arnold data
                        is_arnold_result, similarity, analysis = is_real_arnold(face_data['embedding'], arnold_embeddings)
                        
                        # Display face and results
                        col1, col2 = st.columns([1, 2])
                        
                        with col1:
                            # Create face crop for display
                            x, y, w, h = face_data['box']
                            face_crop = image_rgb[y:y+h, x:x+w]
                            st.image(Image.fromarray(face_crop), width=120, caption=f"Face {i+1}")
                        
                        with col2:
                            if is_arnold_result:
                                st.markdown(f"""
                                <div class="result-box arnold-result">
                                    <h3>🏋️ Face {i+1}: Arnold Schwarzenegger</h3>
                                    <p><strong>Similarity:</strong> {similarity:.1%}</p>
                                    <p><strong>Confidence:</strong> {face_data['confidence']:.1%}</p>
                                    <p><strong>Analysis:</strong> {analysis}</p>
                                </div>
                                """, unsafe_allow_html=True)
                            else:
                                st.markdown(f"""
                                <div class="result-box not-arnold-result">
                                    <h3>👤 Face {i+1}: Not Arnold</h3>
                                    <p><strong>Similarity:</strong> {similarity:.1%}</p>
                                    <p><strong>Confidence:</strong> {face_data['confidence']:.1%}</p>
                                    <p><strong>Analysis:</strong> {analysis}</p>
                                </div>
                                """, unsafe_allow_html=True)
                    
                    # Summary metrics
                    arnold_count = sum(1 for face_data in face_results if is_real_arnold(face_data['embedding'], arnold_embeddings)[0])
                    total_faces = len(face_results)
                    
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
                    st.error(f"❌ Could not analyze image: {message}")

# Instructions
st.markdown("---")
st.markdown("""
### 📋 How It Works:
1. **Loads Arnold images** from dataset
2. **Extracts face embeddings** from photos
3. **Compares uploaded face** to reference data
4. **Uses similarity matching** for detection

### 💡 System Info:
- **Data Source:** First 30 Arnold images (speed optimized)
- **Method:** Cosine similarity comparison
- **Threshold:** Adaptive (28-35% based on confidence)
- **Models:** FaceNet512 (optimized for consistency)
""")

# Footer
st.markdown("---")
st.markdown('<p style="text-align: center; color: #6c757d;">🏋️ Arnold Facial Recognition System</p>', unsafe_allow_html=True)
