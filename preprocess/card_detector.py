import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s"
)
logger = logging.getLogger(__name__)

# ============================================================
# Defaults & Constants (Exported for backward compatibility)
# ============================================================

INPUT_PATH = Path("national_id.jpg")
OUTPUT_DIR = Path("debug_card_detection")

class CardDetectorConfig:
    """Configuration for card detection parameters."""
    MAX_SIDE = 1600
    BORDER_RATIO = 0.04
    MIN_ASPECT_RATIO = 1.20
    MAX_ASPECT_RATIO = 2.20
    MIN_AREA_RATIO = 0.20
    MAX_AREA_RATIO = 0.95
    GRABCUT_ITERATIONS = 8
    NORMALIZED_WIDTH = 1400
    NORMALIZED_HEIGHT = 840
    CORNER_EPSILON_RATIOS = (0.01, 0.015, 0.02, 0.025, 0.03, 0.04, 0.05)

# ============================================================
# Utilities
# ============================================================

def order_corners(points: np.ndarray) -> np.ndarray:
    """
    Order corners in the sequence: [top-left, top-right, bottom-right, bottom-left].
    """
    points = np.asarray(points, dtype=np.float32)
    ordered = np.zeros((4, 2), dtype=np.float32)

    sums = points.sum(axis=1)
    ordered[0] = points[np.argmin(sums)]  # top-left has min sum
    ordered[2] = points[np.argmax(sums)]  # bottom-right has max sum

    diffs = points[:, 1] - points[:, 0]
    ordered[1] = points[np.argmin(diffs)] # top-right has min difference (y - x)
    ordered[3] = points[np.argmax(diffs)] # bottom-left has max difference (y - x)

    return ordered

# ============================================================
# Core Detection Logic
# ============================================================

class CardDetector:
    def __init__(self, config: CardDetectorConfig = CardDetectorConfig()):
        self.config = config

    def resize_for_detection(self, image: np.ndarray) -> Tuple[np.ndarray, float]:
        """Resize image for faster detection while maintaining aspect ratio."""
        h, w = image.shape[:2]
        largest_side = max(h, w)

        if largest_side <= self.config.MAX_SIDE:
            return image.copy(), 1.0

        scale = self.config.MAX_SIDE / largest_side
        resized = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        return resized, scale

    def run_grabcut(self, image: np.ndarray) -> np.ndarray:
        """Run GrabCut to segment foreground based on image borders."""
        h, w = image.shape[:2]
        mask = np.full((h, w), cv2.GC_PR_FGD, dtype=np.uint8)
        
        border = max(5, int(min(h, w) * self.config.BORDER_RATIO))
        mask[:border, :] = cv2.GC_BGD
        mask[-border:, :] = cv2.GC_BGD
        mask[:, :border] = cv2.GC_BGD
        mask[:, -border:] = cv2.GC_BGD

        bgd_model = np.zeros((1, 65), np.float64)
        fgd_model = np.zeros((1, 65), np.float64)

        cv2.grabCut(image, mask, None, bgd_model, fgd_model, self.config.GRABCUT_ITERATIONS, cv2.GC_INIT_WITH_MASK)
        
        return np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)

    def clean_mask(self, mask: np.ndarray) -> np.ndarray:
        """Apply morphological operations to clean up the GrabCut mask."""
        h, w = mask.shape
        kernel_size = max(5, int(min(h, w) * 0.015))
        if kernel_size % 2 == 0: kernel_size += 1

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1) # Remove small noise
        return mask

    def get_components(self, mask: np.ndarray) -> Tuple[List[Dict], np.ndarray]:
        """Extract connected components and their statistics from the mask."""
        h, w = mask.shape
        total_area = float(h * w)
        n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

        components = []
        for i in range(1, n_labels):
            x, y, cw, ch, area = stats[i]
            if cw <= 0 or ch <= 0: continue

            components.append({
                "label": i, "x": x, "y": y, "width": cw, "height": ch, "area": area,
                "area_ratio": area / total_area,
                "ratio": cw / ch,
                "rectangularity": area / (cw * ch) if cw * ch > 0 else 0
            })
        return components, labels

    def score_candidate(self, candidate: Dict, img_shape: Tuple[int, int]) -> float:
        """Score a candidate component based on geometry and position."""
        h, w = img_shape[:2]
        score = 0.0

        # Area score
        if self.config.MIN_AREA_RATIO <= candidate["area_ratio"] <= self.config.MAX_AREA_RATIO:
            score += 35
        else:
            dist = max(0, self.config.MIN_AREA_RATIO - candidate["area_ratio"], candidate["area_ratio"] - self.config.MAX_AREA_RATIO)
            score -= min(30, dist * 100)

        # Aspect ratio score
        if self.config.MIN_ASPECT_RATIO <= candidate["ratio"] <= self.config.MAX_ASPECT_RATIO:
            score += 35
            target_ratio = 1.59
            score += max(0, 15 - abs(candidate["ratio"] - target_ratio) * 15)
        else:
            score -= 30

        # Rectangularity
        score += min(20, candidate["rectangularity"] * 20)

        # Size and position
        if candidate["width"] / w > 0.50: score += 5
        if candidate["height"] / h > 0.40: score += 5

        # Border penalty
        touches = [
            candidate["x"] <= w * 0.01,
            candidate["y"] <= h * 0.01,
            candidate["x"] + candidate["width"] >= w * 0.99,
            candidate["y"] + candidate["height"] >= h * 0.99
        ]
        score -= sum(touches) * 15

        return score

    def find_best_candidate(self, components: List[Dict], img_shape: Tuple[int, int]) -> List[Dict]:
        """Rank and return candidates by score."""
        for c in components:
            c["score"] = self.score_candidate(c, img_shape)
        return sorted(components, key=lambda x: x["score"], reverse=True)

    def find_corners(self, labels: np.ndarray, candidate: Dict) -> np.ndarray:
        """Extract the 4 corners of the candidate using contour approximation."""
        mask = np.where(labels == candidate["label"], 255, 0).astype(np.uint8)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return self._fallback_corners(candidate)

        contour = max(contours, key=cv2.contourArea)
        peri = cv2.arcLength(contour, True)

        for eps in self.config.CORNER_EPSILON_RATIOS:
            approx = cv2.approxPolyDP(contour, eps * peri, True)
            if len(approx) == 4 and cv2.isContourConvex(approx):
                return order_corners(approx.reshape(4, 2))

        # Fallback to rotated bounding box
        rect = cv2.minAreaRect(contour)
        return order_corners(cv2.boxPoints(rect))

    def _fallback_corners(self, candidate: Dict) -> np.ndarray:
        """Axis-aligned fallback corners."""
        x, y, w, h = candidate["x"], candidate["y"], candidate["width"], candidate["height"]
        corners = np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]], dtype=np.float32)
        return order_corners(corners)

    def detect(self, image: np.ndarray) -> np.ndarray:
        """Run the full detection pipeline and return ordered corners."""
        orig_h, orig_w = image.shape[:2]
        working_img, scale = self.resize_for_detection(image)

        mask = self.run_grabcut(working_img)
        mask = self.clean_mask(mask)
        components, labels = self.get_components(mask)

        candidates = self.find_best_candidate(components, working_img.shape)
        if not candidates or candidates[0]["score"] < 30:
            raise RuntimeError("Could not confidently detect the ID card.")

        corners = self.find_corners(labels, candidates[0])
        
        if scale != 1.0:
            corners /= scale

        corners[:, 0] = np.clip(corners[:, 0], 0, orig_w - 1)
        corners[:, 1] = np.clip(corners[:, 1], 0, orig_h - 1)
        
        return corners

# ============================================================
# API Functions
# ============================================================

def detect_card(image: np.ndarray) -> np.ndarray:
    """Convenience wrapper for CardDetector.detect()."""
    return CardDetector().detect(image)

def perspective_correct(image: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Warp the detected region into a top-down view."""
    pts = order_corners(points)
    tl, tr, br, bl = pts
    
    w1 = np.linalg.norm(br - bl)
    w2 = np.linalg.norm(tr - tl)
    h1 = np.linalg.norm(tr - br)
    h2 = np.linalg.norm(tl - bl)
    
    max_w = max(1, int(round(max(w1, w2))))
    max_h = max(1, int(round(max(h1, h2))))

    dst = np.array([[0, 0], [max_w - 1, 0], [max_w - 1, max_h - 1], [0, max_h - 1]], dtype=np.float32)
    matrix = cv2.getPerspectiveTransform(pts, dst)
    return cv2.warpPerspective(image, matrix, (max_w, max_h))

def normalize_card(image: np.ndarray) -> np.ndarray:
    """Resize to a standard fixed size for consistent field extraction."""
    return cv2.resize(image, (CardDetectorConfig.NORMALIZED_WIDTH, CardDetectorConfig.NORMALIZED_HEIGHT), interpolation=cv2.INTER_AREA)

# ============================================================
# Debug Drawing
# ============================================================

def draw_debug_info(image: np.ndarray, corners: np.ndarray) -> np.ndarray:
    """Draw corner labels and boundary polygon for debugging."""
    output = image.copy()
    pts = corners.astype(np.int32)
    cv2.polylines(output, [pts], True, (0, 255, 0), 5)
    
    labels = ["TL", "TR", "BR", "BL"]
    for i, (p, label) in enumerate(zip(pts, labels)):
        cv2.circle(output, tuple(p), 12, (0, 0, 255), -1)
        cv2.putText(output, f"{i}: {label}", (p[0] + 15, p[1] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    return output

# ============================================================
# CLI Entry Point
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Egyptian National ID Card Detector")
    parser.add_argument("--input", type=str, default=str(INPUT_PATH), help="Path to input image")
    parser.add_argument("--output", type=str, default=str(OUTPUT_DIR), help="Directory for debug output")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output)

    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        return

    image = cv2.imread(str(input_path))
    if image is None:
        logger.error("Could not read image.")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        detector = CardDetector()
        corners = detector.detect(image)
        
        logger.info("Card detected successfully.")
        
        # Save debug image
        debug_img = draw_debug_info(image, corners)
        cv2.imwrite(str(output_dir / "card_detected.jpg"), debug_img)
        
        # Save corrected/normalized images
        corrected = perspective_correct(image, corners)
        cv2.imwrite(str(output_dir / "card_corrected.jpg"), corrected)
        
        normalized = normalize_card(corrected)
        cv2.imwrite(str(output_dir / "card_normalized.jpg"), normalized)
        
        logger.info(f"Results saved to: {output_dir}")

    except Exception as e:
        logger.exception(f"Detection failed: {e}")

if __name__ == "__main__":
    main()
