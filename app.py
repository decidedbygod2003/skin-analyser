import streamlit as st
import tempfile
import json
import cv2
import numpy as np
import os
import logging

from analyze import (
    GeminiClient,
    DetectionNormalizer,
    ImageAnnotator,
    StatisticsGenerator,
    get_analysis_prompt
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --------------------------------------------------
# PAGE CONFIG (MUST BE FIRST STREAMLIT CALL)
# --------------------------------------------------
st.set_page_config(
    page_title="Clinical Skin Analysis",
    layout="wide"
)

st.title("🧬 Clinical Skin Concern Detection")
st.caption("Dermatology-grade analysis using Google Gemini Vision AI")

# --------------------------------------------------
# API KEY CHECK
# --------------------------------------------------
if not os.getenv("GOOGLE_API_KEY"):
    st.error("⚠️ GOOGLE_API_KEY not found in environment variables!")
    st.info("Please add your Google API key to continue.")
    st.stop()

# --------------------------------------------------
# SETTINGS SIDEBAR
# --------------------------------------------------
with st.sidebar:
    st.header("⚙️ Settings")
    
    show_debug = st.checkbox("Show Debug Info", value=False)
    use_custom_prompt = st.checkbox("Use Custom Prompt", value=False)
    
    min_confidence = st.slider(
        "Minimum Confidence Threshold",
        min_value=0,
        max_value=100,
        value=40,
        help="Only show detections above this confidence level"
    )

# --------------------------------------------------
# PROMPT CONFIGURATION
# --------------------------------------------------
st.subheader("🧠 Analysis Configuration")

if use_custom_prompt:
    prompt = st.text_area(
        "Custom Analysis Prompt",
        value=get_analysis_prompt(False),
        height=400,
        help="Advanced users can modify the prompt for different analysis styles"
    )
else:
    st.info("Using optimized default prompt. Enable 'Use Custom Prompt' in sidebar to customize.")
    prompt = get_analysis_prompt(False)

# --------------------------------------------------
# IMAGE SOURCE
# --------------------------------------------------
st.subheader("📸 Upload or Capture Face Image")

image_source = st.radio(
    "Choose image source:",
    ["Upload Image", "Capture from Camera"],
    horizontal=True
)

image_path = None

# --------------------------------------------------
# IMAGE PREPROCESSING & VALIDATION
# --------------------------------------------------
def preprocess_camera_image(path: str):
    """Enhance camera image quality"""
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Normalize resolution
    img = cv2.resize(img, (512, 512))
    
    # Enhance brightness and contrast
    img = cv2.convertScaleAbs(img, alpha=1.2, beta=10)

    cv2.imwrite(path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    logger.info(f"Preprocessed camera image: {path}")

def is_face_visible(path: str) -> tuple[bool, str]:
    """Check if face is visible in image"""
    try:
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        img = cv2.imread(path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        
        if len(faces) > 0:
            return True, f"✅ {len(faces)} face(s) detected"
        else:
            return False, "⚠️ No face detected. Please ensure face is clearly visible."
    except Exception as e:
        logger.error(f"Face detection error: {e}")
        return True, "⚠️ Face detection skipped (error occurred)"

# --------------------------------------------------
# IMAGE INPUT
# --------------------------------------------------
if image_source == "Upload Image":
    uploaded = st.file_uploader(
        "Upload a facial image",
        type=["jpg", "jpeg", "png", "webp"],
        help="Select a clear photo of your face with good lighting"
    )

    if uploaded:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            tmp.write(uploaded.read())
            image_path = tmp.name
        logger.info(f"Image uploaded: {image_path}")

if image_source == "Capture from Camera":
    captured = st.camera_input("Capture face image")

    if captured:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            tmp.write(captured.getbuffer())
            image_path = tmp.name
        logger.info(f"Image captured: {image_path}")

        st.info(
            "📸 **Tips for best results:**\n"
            "- Use good natural lighting\n"
            "- Hold camera steady\n"
            "- Keep face centered and close\n"
            "- Avoid strong shadows or glare"
        )

# --------------------------------------------------
# PREVIEW
# --------------------------------------------------
if image_path:
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.image(image_path, caption="Selected Image", use_column_width=True)
    
    with col2:
        # Face validation
        face_found, face_msg = is_face_visible(image_path)
        if face_found:
            st.success(face_msg)
        else:
            st.warning(face_msg)

# --------------------------------------------------
# RUN ANALYSIS
# --------------------------------------------------
if image_path and st.button("🔬 Run Skin Analysis", type="primary", use_container_width=True):

    # Preprocessing for camera images
    if image_source == "Capture from Camera":
        with st.spinner("Preprocessing camera image..."):
            preprocess_camera_image(image_path)
        
        # Use camera-optimized prompt
        prompt = get_analysis_prompt(is_camera=True)

    with st.spinner("🤖 Analyzing skin with Gemini AI... This may take 15-30 seconds..."):
        try:
            # ----------------------------
            # GEMINI ANALYSIS
            # ----------------------------
            gemini = GeminiClient()
            raw_output = gemini.analyze(image_path, prompt)

            if show_debug:
                with st.expander("🔍 Debug: Raw Gemini Output"):
                    st.json(raw_output)

            # ----------------------------
            # NORMALIZE OUTPUT
            # ----------------------------
            detections = DetectionNormalizer.normalize(raw_output)
            
            # Filter by confidence threshold
            filtered_detections = {}
            for label, items in detections.items():
                filtered_items = [item for item in items if item["confidence"] >= min_confidence]
                if filtered_items:
                    filtered_detections[label] = filtered_items
            
            detections = filtered_detections

            if show_debug:
                with st.expander("🔍 Debug: Normalized Detections"):
                    st.json(detections)

            # ----------------------------
            # ANNOTATE IMAGE
            # ----------------------------
            output_image = image_path.replace(".jpg", "_annotated.jpg")
            annotator = ImageAnnotator(image_path)
            annotator.draw(detections)
            annotator.save(output_image)

            # ----------------------------
            # STATS
            # ----------------------------
            stats = StatisticsGenerator.generate(detections)

            st.success("✅ Analysis complete!")

            # --------------------------------------------------
            # OUTPUT DISPLAY
            # --------------------------------------------------
            if not detections:
                st.info("🎉 Great news! No significant skin concerns detected in this image.")
                st.image(image_path, caption="Original Image", use_column_width=True)
            else:
                col1, col2 = st.columns(2)

                with col1:
                    st.subheader("🖼️ Annotated Result")
                    st.image(output_image, use_column_width=True)
                    
                    # Download annotated image
                    with open(output_image, "rb") as f:
                        st.download_button(
                            "⬇️ Download Annotated Image",
                            data=f.read(),
                            file_name="skin_analysis_annotated.jpg",
                            mime="image/jpeg"
                        )

                with col2:
                    st.subheader("📋 Detected Concerns")
                    
                    # Display as cards
                    for label, items in detections.items():
                        with st.container():
                            st.markdown(f"### {label.replace('_', ' ').title()}")
                            for idx, item in enumerate(items, 1):
                                conf = item['confidence']
                                
                                # Color code by confidence
                                if conf >= 70:
                                    emoji = "🔴"
                                elif conf >= 50:
                                    emoji = "🟡"
                                else:
                                    emoji = "🟢"
                                
                                st.markdown(f"{emoji} **Detection {idx}**: {conf:.1f}% confidence")
                            st.divider()

                # Statistics
                if stats:
                    st.subheader("📊 Summary Statistics")
                    
                    # Create a more visual stats display
                    stats_cols = st.columns(len(stats))
                    for idx, (label, data) in enumerate(stats.items()):
                        with stats_cols[idx]:
                            st.metric(
                                label=label.replace('_', ' ').title(),
                                value=f"{data['count']} area(s)",
                                delta=f"{data['avg_confidence']:.0f}% avg"
                            )

                # JSON Export
                with st.expander("📄 View JSON Data"):
                    st.json({
                        "detections": detections,
                        "statistics": stats
                    })

                st.download_button(
                    "⬇️ Download Analysis Results (JSON)",
                    data=json.dumps(
                        {
                            "detections": detections,
                            "statistics": stats,
                            "settings": {
                                "min_confidence": min_confidence,
                                "image_source": image_source
                            }
                        },
                        indent=2
                    ),
                    file_name="skin_analysis_results.json",
                    mime="application/json"
                )

        except Exception as e:
            st.error(f"❌ Analysis failed: {str(e)}")
            logger.error(f"Analysis error: {e}", exc_info=True)
            
            if show_debug:
                st.exception(e)
            
            st.info(
                "**Troubleshooting tips:**\n"
                "- Ensure your Google API key is valid\n"
                "- Try a different image with better lighting\n"
                "- Enable 'Show Debug Info' in sidebar for more details"
            )

# --------------------------------------------------
# FOOTER
# --------------------------------------------------
st.divider()
st.caption("Powered by Google Gemini AI • Built for Unfilter Platform")