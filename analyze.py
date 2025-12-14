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
        "fine lines",
        "dark_circles",
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
    
    base = """You are a professional dermatology AI analyzing facial skin.

CRITICAL INSTRUCTIONS:
1. Analyze the face for ONLY these skin concerns:
   - acne (pimples, blemishes, breakouts)
   - oiliness (shiny areas, excessive sebum)
   - wrinkles (deep lines, creases)
   - fine_lines (subtle lines, early aging signs)
   - dark_circles (under-eye darkness, shadows)
   - redness (inflammation, irritation, uneven tone)
   - pores (enlarged, visible pores)
   - dryness (flaky, rough texture)
   - pigmentation (dark spots, melasma, uneven tone)

2. For EACH detected concern, you MUST provide:
   - confidence: 0-100 (how certain you are)
   - box: bounding box coordinates

3. Bounding box format (REQUIRED):
   {
     "x_min": 0-100,  // left edge as % of image width
     "y_min": 0-100,  // top edge as % of image height
     "x_max": 0-100,  // right edge as % of image width
     "y_max": 0-100   // bottom edge as % of image height
   }

4. Output format MUST be valid JSON:
   {
     "concern_name": [
       {
         "confidence": 75,
         "box": {"x_min": 30, "y_min": 40, "x_max": 70, "y_max": 60}
       }
     ]
   }

5. Rules:
   - If multiple areas show same concern, create separate entries
   - Use approximate bounding boxes for widespread concerns
   - Return {} if no concerns detected
   - NO markdown, NO explanations, ONLY JSON
"""

    if is_camera:
        base += """

CAMERA IMAGE NOTES:
- Accept lower confidence (40%+) for subtle concerns
- Account for lighting variations
- Be more lenient with detection
"""

    return base