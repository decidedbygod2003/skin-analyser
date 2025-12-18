 
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import tempfile
import os
import base64
import logging
 
from analyze import (
    GeminiClient,
    DetectionNormalizer,
    ImageAnnotator,
    StatisticsGenerator,
    get_analysis_prompt
)
 
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
 
app = FastAPI(
    title="Clinical Skin Analysis API",
    version="1.0"
)
 
# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
 
 
def encode_image_to_base64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")
 
 
@app.get("/")
async def root():
    return {"message": "Clinical Skin Analysis API", "status": "running"}
 
 
@app.get("/health")
async def health():
    return {"status": "healthy"}
 
 
@app.post("/analyze-skin")
async def analyze_skin(image: UploadFile = File(...)):
    if image.content_type not in ["image/jpeg", "image/png", "image/jpg", "image/webp"]:
        raise HTTPException(status_code=400, detail="Invalid image format")
 
    if not os.getenv("GOOGLE_API_KEY"):
        raise HTTPException(status_code=500, detail="GOOGLE_API_KEY missing")
 
    try:
        # ----------------------------
        # Save uploaded image
        # ----------------------------
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            tmp.write(await image.read())
            image_path = tmp.name
 
        # ----------------------------
        # Gemini Analysis
        # ----------------------------
        gemini = GeminiClient()
        prompt = get_analysis_prompt(is_camera=False)
        raw_output = gemini.analyze(image_path, prompt)
 
        # ----------------------------
        # Normalize
        # ----------------------------
        detections = DetectionNormalizer.normalize(raw_output)
 
        # ----------------------------
        # Annotate Image
        # ----------------------------
        annotated_path = image_path.replace(".jpg", "_annotated.jpg")
        annotator = ImageAnnotator(image_path)
        annotator.draw(detections)
        annotator.save(annotated_path)
 
        # ----------------------------
        # Stats
        # ----------------------------
        stats = StatisticsGenerator.generate(detections)
 
        # ----------------------------
        # Encode Image
        # ----------------------------
        annotated_b64 = encode_image_to_base64(annotated_path)
 
        # Cleanup
        os.remove(image_path)
        os.remove(annotated_path)
 
        return JSONResponse({
            "status": "success",
            "detections": detections,
            "statistics": stats,
            "annotated_image_base64": annotated_b64
        })
 
    except Exception as e:
        logger.error(f"Skin analysis failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
 
 