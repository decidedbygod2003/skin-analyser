"""
Skin Concern Detection using Google Gemini AI
Professional dermatology-grade skin analysis with improved bounding box detection
"""
 
import json
import logging
import os
from dataclasses import dataclass
from typing import Dict, List, Optional
 
import cv2
import google.generativeai as genai
from dotenv import load_dotenv
from PIL import Image
 
# --------------------------------------------------
# LOGGING
# --------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
load_dotenv()
 
# --------------------------------------------------
# CONFIG
# --------------------------------------------------
@dataclass
class Config:
    ALLOWED_LABELS = {
        "acne",
        "oiliness",
        "wrinkles",
        "finelines",
        "dark circles",
        "redness",
        "pores",
        "dryness",
        "pigmentation",
    }
 
    LABEL_COLORS = {
        "acne": (0, 0, 255),
        "oiliness": (0, 255, 255),
        "wrinkles": (255, 0, 0),
        "finelines": (255, 100, 0),
        "fine lines": (255, 100, 0),
        "dark_circles": (128, 0, 128),
        "dark circles": (128, 0, 128),
        "redness": (0, 165, 255),
        "pores": (0, 255, 0),
        "dryness": (255, 255, 255),
        "pigmentation": (139, 69, 19),
    }
 
    MODEL_NAME = "gemini-2.5-flash"
 
# --------------------------------------------------
# GEMINI CLIENT WITH IMPROVED PROMPTING
# --------------------------------------------------
class GeminiClient:
    def __init__(self):
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError("GOOGLE_API_KEY missing from environment")
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(Config.MODEL_NAME)
 
    @staticmethod
    def clean_json(text: str) -> Dict:
        text = text.strip()
        # Remove markdown code blocks
        if text.startswith("```"):
            text = text.replace("```json", "").replace("```", "").strip()
        return json.loads(text)
 
    def analyze(self, image_path: str, prompt: str) -> Dict:
        """Analyze image with Gemini and return structured JSON"""
        image = Image.open(image_path)
       
        try:
            response = self.model.generate_content([prompt, image])
           
            # Log raw response for debugging
            logger.info(f"Raw Gemini response (first 800 chars): {response.text[:800]}")
           
            parsed = self.clean_json(response.text)
            logger.info(f"Parsed JSON structure: {json.dumps(parsed, indent=2)[:500]}")
           
            return parsed
           
        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing failed: {e}")
            logger.error(f"Full response text: {response.text}")
            raise ValueError(f"Gemini returned invalid JSON: {e}")
        except Exception as e:
            logger.error(f"Analysis failed: {e}")
            raise
 
# --------------------------------------------------
# IMPROVED DETECTION NORMALIZER
# --------------------------------------------------
class DetectionNormalizer:
    @staticmethod
    def normalize(raw: Dict) -> Dict:
        """
        Normalizes Gemini output into standardized format.
        Handles multiple box formats and creates default boxes if missing.
       
        Returns:
        {
          "label": [
            {
              "confidence": float,
              "box_pct": {"x_min": float, "y_min": float, "x_max": float, "y_max": float}
            }
          ]
        }
        """
        normalized = {}
 
        for label, items in raw.items():
            label_l = label.lower().strip().replace("_", " ")
 
            # Check if label is allowed
            if label_l not in Config.ALLOWED_LABELS and label_l.replace(" ", "_") not in Config.ALLOWED_LABELS:
                logger.warning(f"Skipping unknown label: {label}")
                continue
 
            clean_items = []
           
            # Handle if items is not a list
            if not isinstance(items, list):
                items = [items]
 
            for idx, d in enumerate(items):
                # Handle different data structures
                if isinstance(d, dict):
                    conf = d.get("confidence", d.get("severity", 50))
                    box = d.get("box") or d.get("bbox") or d.get("box_2d") or d.get("location")
                   
                    # If no confidence provided, use default
                    if conf is None:
                        conf = 50
                        logger.warning(f"No confidence found for {label}, using default: 50")
                   
                    # If no box provided, create default box for this concern area
                    if box is None:
                        logger.warning(f"No bounding box found for {label}, creating default box")
                        box = DetectionNormalizer._create_default_box(label_l, idx)
                   
                    # Normalize box format
                    normalized_box = DetectionNormalizer._normalize_box(box)
                   
                    if normalized_box:
                        clean_items.append({
                            "confidence": float(conf),
                            "box_pct": normalized_box
                        })
                else:
                    # If detection is just a string or number, create default entry
                    logger.warning(f"Unexpected format for {label}: {d}")
                    clean_items.append({
                        "confidence": 50,
                        "box_pct": DetectionNormalizer._create_default_box(label_l, idx)
                    })
 
            if clean_items:
                normalized[label_l] = clean_items
                logger.info(f"Normalized {len(clean_items)} detections for '{label_l}'")
 
        return normalized
 
    @staticmethod
    def _normalize_box(box: Dict) -> Optional[Dict]:
        """Convert various box formats to percentage-based x_min, y_min, x_max, y_max"""
       
        # Format 1: x_min, y_min, x_max, y_max (percentage)
        if all(k in box for k in ["x_min", "y_min", "x_max", "y_max"]):
            return {
                "x_min": float(box["x_min"]),
                "y_min": float(box["y_min"]),
                "x_max": float(box["x_max"]),
                "y_max": float(box["y_max"])
            }
       
        # Format 2: x, y, width, height (percentage)
        if all(k in box for k in ["x", "y", "width", "height"]):
            return {
                "x_min": float(box["x"]),
                "y_min": float(box["y"]),
                "x_max": float(box["x"]) + float(box["width"]),
                "y_max": float(box["y"]) + float(box["height"])
            }
       
        # Format 3: top, left, bottom, right
        if all(k in box for k in ["top", "left", "bottom", "right"]):
            return {
                "x_min": float(box["left"]),
                "y_min": float(box["top"]),
                "x_max": float(box["right"]),
                "y_max": float(box["bottom"])
            }
       
        logger.warning(f"Unknown box format: {box}")
        return None
 
    @staticmethod
    def _create_default_box(label: str, index: int = 0) -> Dict:
        """Create default bounding boxes based on typical face regions"""
       
        # Default boxes for common facial regions (percentage-based)
        default_boxes = {
            "acne": {"x_min": 25, "y_min": 25, "x_max": 75, "y_max": 65},  # Full face
            "oiliness": {"x_min": 30, "y_min": 20, "x_max": 70, "y_max": 50},  # T-zone
            "wrinkles": {"x_min": 20, "y_min": 35, "x_max": 80, "y_max": 55},  # Eye/forehead area
            "fine lines": {"x_min": 25, "y_min": 35, "x_max": 75, "y_max": 50},  # Eye area
            "finelines": {"x_min": 25, "y_min": 35, "x_max": 75, "y_max": 50},
            "dark circles": {"x_min": 30, "y_min": 40, "x_max": 70, "y_max": 55},  # Under eyes
            "dark_circles": {"x_min": 30, "y_min": 40, "x_max": 70, "y_max": 55},
            "redness": {"x_min": 35, "y_min": 35, "x_max": 65, "y_max": 55},  # Cheeks
            "pores": {"x_min": 30, "y_min": 30, "x_max": 70, "y_max": 60},  # Nose/cheek area
            "dryness": {"x_min": 25, "y_min": 30, "x_max": 75, "y_max": 70},  # Full face
            "pigmentation": {"x_min": 30, "y_min": 30, "x_max": 70, "y_max": 60},  # Face area
        }
       
        # Return default box for this label, with slight offset if multiple detections
        default = default_boxes.get(label, {"x_min": 25, "y_min": 25, "x_max": 75, "y_max": 75})
       
        # Add slight offset for multiple detections of same type
        if index > 0:
            offset = index * 5
            default = {k: v + offset if "min" in k else v - offset for k, v in default.items()}
       
        return default
 
# --------------------------------------------------
# IMAGE ANNOTATOR
# --------------------------------------------------
class ImageAnnotator:
    def __init__(self, image_path: str):
        self.image = cv2.imread(image_path)
        if self.image is None:
            raise ValueError(f"Failed to load image: {image_path}")
        self.h, self.w, _ = self.image.shape
        logger.info(f"Image loaded: {self.w}x{self.h}")
 
    def _pct_to_px(self, b: Dict) -> tuple:
        """Convert percentage coordinates to pixels"""
        return (
            int(b["x_min"] / 100 * self.w),
            int(b["y_min"] / 100 * self.h),
            int(b["x_max"] / 100 * self.w),
            int(b["y_max"] / 100 * self.h),
        )
 
    def draw(self, detections: Dict):
        """Draw bounding boxes and labels on image"""
        if not detections:
            logger.warning("No detections to draw")
            # Add overlay text indicating no detections
            cv2.putText(
                self.image,
                "No skin concerns detected",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 0),
                2
            )
            return
 
        for label, items in detections.items():
            color = Config.LABEL_COLORS.get(label, (255, 255, 255))
            logger.info(f"Drawing {len(items)} boxes for '{label}'")
 
            for idx, d in enumerate(items):
                box_pct = d.get("box_pct")
                if not box_pct:
                    logger.warning(f"No box_pct found for {label}")
                    continue
 
                try:
                    x1, y1, x2, y2 = self._pct_to_px(box_pct)
                   
                    # Ensure coordinates are within image bounds
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(self.w, x2), min(self.h, y2)
                   
                    # Draw rectangle
                    cv2.rectangle(self.image, (x1, y1), (x2, y2), color, 3)
                   
                    # Draw label with background
                    label_text = f"{label.replace('_', ' ').title()}: {d['confidence']:.0f}%"
                   
                    # Get text size for background
                    (text_w, text_h), baseline = cv2.getTextSize(
                        label_text,
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        2
                    )
                   
                    # Draw background rectangle for text
                    cv2.rectangle(
                        self.image,
                        (x1, max(0, y1 - text_h - 10)),
                        (x1 + text_w, y1),
                        color,
                        -1  # Filled
                    )
                   
                    # Draw text
                    cv2.putText(
                        self.image,
                        label_text,
                        (x1, max(text_h, y1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255, 255, 255),  # White text
                        2
                    )
                   
                    logger.info(f"Drew box for {label} at ({x1},{y1})-({x2},{y2})")
                   
                except Exception as e:
                    logger.error(f"Failed to draw box for {label}: {e}")
                    continue
 
    def save(self, path: str):
        """Save annotated image"""
        success = cv2.imwrite(path, self.image)
        if success:
            logger.info(f"Saved annotated image to: {path}")
        else:
            logger.error(f"Failed to save image to: {path}")
 
# --------------------------------------------------
# STATISTICS
# --------------------------------------------------
class StatisticsGenerator:
    @staticmethod
    def generate(detections: Dict) -> Dict:
        """Generate summary statistics from detections"""
        if not detections:
            return {}
           
        stats = {}
        for label, items in detections.items():
            if items:
                stats[label] = {
                    "count": len(items),
                    "avg_confidence": round(
                        sum(i["confidence"] for i in items) / len(items), 2
                    ),
                    "max_confidence": round(max(i["confidence"] for i in items), 2),
                    "min_confidence": round(min(i["confidence"] for i in items), 2)
                }
        return stats
 
 
# --------------------------------------------------
# IMPROVED PROMPT GENERATOR
# --------------------------------------------------
def get_analysis_prompt(is_camera: bool = False) -> str:
    """Generate optimized prompt for Gemini"""
   
    base = """You are a professional dermatology-grade skin analysis AI trained in
clinical dermatology, age-related skin physiology, and ethnodermatology.
 
Your task is to analyze the provided facial image and detect ONLY the following allowed skin concerns:
 
{labels_str}
 
You MUST return ONLY a strictly valid JSON object.
DO NOT include any text, explanation, markdown, or comments before or after the JSON.

═══════════════════════════════════════════════════════════
GLOBAL OUTPUT RULES (STRICT – NON-NEGOTIABLE)
═══════════════════════════════════════════════════════════

1. CONFIDENCE MUST BE A NUMBER (float or integer).
   ✗ DO NOT use words such as: "low", "medium", "high", "moderate".
   ✓ Allowed: numeric values ONLY (40–95).

2. DO NOT assign MEDIUM confidence (60–79%) unless
   ALL clinical criteria for that condition are clearly met.

3. If criteria are NOT clearly met:
   ✓ Either assign LOW confidence (40–59%)
   ✓ OR do NOT detect the concern at all.

4. When uncertain, prefer:
   → NON-DETECTION ({})
   → rather than an unjustified MEDIUM confidence.

5. NEVER inflate confidence to fill output.
   Returning {} is always acceptable and clinically correct.
 
═══════════════════════════════════════════════════════════
STEP 1 — VISUAL CONTEXT ESTIMATION (INTERNAL ONLY)
═══════════════════════════════════════════════════════════
 
Before detecting skin concerns, you MUST internally estimate the following
based ONLY on visible skin characteristics (do NOT output these estimates):
 
A. APPROXIMATE AGE GROUP
   - Teenager (≤19)
   - Young Adult (25–35)
   - Adult (36–50)
   - Mature (51+)
 
B. DOMINANT SKIN TONE GROUP – Use FitPatrick Scale
  - Light / Fair
   - Medium / Olive
   - Brown
   - Deep / Dark
 
IMPORTANT:
- Do NOT output age, gender, race, or ethnicity.
- These estimations are used ONLY to adapt diagnostic thresholds.
- Never make cultural, genetic, or identity assumptions.
 
═══════════════════════════════════════════════════════════
STEP 2 — AGE-AWARE DERMATOLOGICAL ADAPTATION
═══════════════════════════════════════════════════════════
 
Adjust detection sensitivity and confidence scoring based on estimated AGE GROUP:
 
▶ TEENAGER (≤19)
✓ Prioritize: ACNE, OILINESS, POST-INFLAMMATORY REDNESS
✓ Allow higher confidence for mild acne lesions
✗ Suppress detection of WRINKLES and FINE LINES
  (unless pathological scarring or permanent folds)
 
▶ YOUNG ADULT (20–35)
✓ Prioritize: ACNE (including hormonal patterns), PORES, OILINESS
✓ Allow EARLY FINE LINES
✓ Detect pigmentation mainly as PIH or early, melasma or Sunspots
✓ Wrinkles most likely will be shallow to qualify
 
▶ ADULT (36–50)
✓ Prioritize: FINE LINES, EARLY WRINKLES, PIGMENTATION
✓ Reduce acne confidence unless clearly inflammatory
✓ DRYNESS becomes clinically relevant
 
▶ MATURE (51+)
✓ Prioritize: WRINKLES (static rhytides), FINE LINES, Mature Skin DRYNESS, PIGMENTATION
✓ ACNE and OILINESS require strong visual evidence
✓ Do NOT penalize normal age-related laxity
 
═══════════════════════════════════════════════════════════
STEP 3 — SKIN-TONE & ETHNODERMATOLOGY ADAPTATION
═══════════════════════════════════════════════════════════
 
Adjust visual interpretation based on estimated SKIN TONE:
 
▶ MEDIUM / BROWN / DEEP SKIN TONES (FitzPatrick III-VI)
(Common in Indian, South Asian, African, Middle Eastern populations)
 
✓ Increase sensitivity for:
  - PIGMENTATION (PIH, melasma, uneven tone)
  - DARK CIRCLES (brown/grey)
✓ REDNESS may appear as:
  - Violaceous, purplish, deep brown warmth
✓ DRYNESS may present as:
  - Ashy, grey, dull texture
✗ Do NOT under-detect redness due to lack of pink hue
 
▶ LIGHT / FAIR SKIN TONES (Fitzpatrick I-III)
(Common in European, North American, Caucasian populations)
 
✓ Increase sensitivity for:
  - REDNESS (erythema, flushing)
  - TELANGIECTASIA
✓ Pigmentation must show clear contrast
✓ Dark circles may appear blue or purple
 
═══════════════════════════════════════════════════════════
STEP 4 — CLINICAL DETECTION GUIDELINES
═══════════════════════════════════════════════════════════
 
Use strict, evidence-based dermatological criteria for each condition:
 
1. ACNE (Acne Vulgaris)
- Inflammatory papules, pustules, nodules, cysts
- Comedones (open or closed)
- Red to purple coloration, 1–5mm typical size
- Typical distribution: cheeks, forehead, chin, jawline
 
2. OILINESS (Seborrhea)
- Specular highlights, greasy or glossy appearance
- T-zone predominance
- Often associated with enlarged pores
 
3. WRINKLES (Deep Rhytides)
- Static, deep creases visible at rest
- Depth >1mm with shadowing
- Forehead, glabellar, periorbital, nasolabial regions
 
4. FINE LINES (Superficial Rhytides)
- Thin, shallow epidermal creases (<0.5mm)
- Early photoaging signs
- Periocular, forehead, perioral regions
 
5. DARK CIRCLES (Periorbital Hyperpigmentation)
- Brown, purple, or bluish discoloration
- 10–15% contrast minimum from surrounding skin
- Under-eye distribution, typically bilateral
 
6. REDNESS (Erythema & Telangiectasia)
- Diffuse or patchy pink/red/violaceous coloration
- Visible dilated capillaries
- Clear contrast with baseline skin tone
 
7. PORES (Enlarged Follicular Ostia)
- Visible follicular openings (>0.25mm)
- Stippled or crater-like texture
- T-zone and medial cheeks
 
8. DRYNESS (Xerosis)
- Flaking, scaling, rough or dull texture
- Loss of natural sheen
- Crepey appearance may be present
 
9. PIGMENTATION (Dyschromia)
- Brown to dark macules or patches
- PIH, melasma, lentigines, freckles
- Must show 15–20% contrast from baseline skin
 
═══════════════════════════════════════════════════════════
STEP 4 — CLINICAL DETECTION THRESHOLDS
═══════════════════════════════════════════════════════════

Use STRICT criteria:

LOW CONFIDENCE (40–59%)
- Early, subtle, or ambiguous findings
- Limited clarity
- Mild texture change

MEDIUM CONFIDENCE (60–79%)
- MULTIPLE confirming visual signs
- Typical anatomical distribution
- Clear visibility at rest
- Adequate lighting and resolution

HIGH CONFIDENCE (80–95%)
- Classic textbook presentation
- Clear borders and contrast
- Unambiguous clinical evidence

⚠️ If MEDIUM criteria are not fully met → downgrade to LOW or suppress.

═══════════════════════════════════════════════════════════
STEP 5 — BOUNDING BOX REQUIREMENTS
═══════════════════════════════════════════════════════════

For EACH detected concern:
✓ Use percentage coordinates (0–100)
✓ Format:
  {"x_min": float, "y_min": float, "x_max": float, "y_max": float}
✓ Boxes must be tight and specific
✓ Multiple regions require multiple boxes
✓ Ensure x_min < x_max and y_min < y_max
 
═══════════════════════════════════════════════════════════
STRICT EXCLUSION CRITERIA
═══════════════════════════════════════════════════════════
 
✗ Makeup or cosmetic effects
✗ Temporary redness, pressure marks, or glare
✗ Hair-obscured or unclear skin regions
✗ Image artifacts or compression noise
✗ Normal anatomy (unless pigmentation-related)
✗ Any condition NOT in {labels_str}
✗ Assumptions based on beauty standards or identity
 
═══════════════════════════════════════════════════════════
OUTPUT FORMAT (STRICT JSON ONLY)
═══════════════════════════════════════════════════════════
 
Return ONLY valid JSON.
 
If NO clinically valid concerns are detected, return:
 
{}
 
 
"""
 
    if is_camera:
        base += """
 
CAMERA IMAGE NOTES:
- Accept lower confidence (40%+) for subtle concerns
- Account for lighting variations
- Be more lenient with detection
"""
 
    return base
 