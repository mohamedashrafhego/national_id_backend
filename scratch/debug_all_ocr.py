
import cv2
import sys
from paddleocr import PaddleOCR
from pathlib import Path

# Add project root to path so we can import preprocess
sys.path.append(str(Path(__file__).parent.parent))

from preprocess.card_detector import (
    INPUT_PATH,
    detect_card,
    perspective_correct,
    normalize_card,
)

def main():
    image = cv2.imread(str(INPUT_PATH))
    corners = detect_card(image)
    corrected = perspective_correct(image, corners)
    normalized = normalize_card(corrected)
    
    cv2.imwrite("debug_normalized_for_ocr.jpg", normalized)
    
    ocr = PaddleOCR(lang="ar", use_angle_cls=True, show_log=False)
    result = ocr.ocr(normalized, cls=True)
    
    print("\n--- ALL DETECTIONS ON NORMALIZED CARD (1400x840) ---")
    for page in result:
        if page is None: continue
        for line in page:
            box = line[0]
            text, score = line[1]
            
            left = min(p[0] for p in box)
            right = max(p[0] for p in box)
            top = min(p[1] for p in box)
            bottom = max(p[1] for p in box)
            
            cx = (left + right) / 2
            cy = (top + bottom) / 2
            
            # Print as ratios for easy copy-paste to card_regions.py
            rx = left / 1400
            ry = top / 840
            rw = (right - left) / 1400
            rh = (bottom - top) / 840
            
            print(f"Text: {text} | Conf: {score:.2f}")
            print(f"  Pixel: [{int(left)}, {int(top)}, {int(right)}, {int(bottom)}] | Center: ({int(cx)}, {int(cy)})")
            print(f"  Ratio: x={rx:.3f}, y={ry:.3f}, w={rw:.3f}, h={rh:.3f}")
            print("-" * 20)

if __name__ == "__main__":
    main()
