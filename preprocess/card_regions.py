"""
Region-of-interest infrastructure for the normalized National ID
card produced by preprocess/card_detector.py.

Regions are defined as ratios (0.0 - 1.0) relative to the
normalized card size, never as absolute pixel coordinates, so
they stay valid regardless of the original photo's resolution.
"""

import cv2
import numpy as np

from preprocess.card_detector import (
    INPUT_PATH,
    OUTPUT_DIR,
    detect_card,
    perspective_correct,
    normalize_card,
)


# ============================================================
# Region configuration
# ============================================================

# Regions matched against the actual normalized-card layout
# (front side of the Egyptian National ID). Coordinates are
# ratios of the 1400 x 840 normalized card, with padding kept
# around each field for OCR tolerance.
CARD_REGIONS = {
    # Face photo, left side of the card.
    "photo": {
        "x": 0.01,
        "y": 0.04,
        "width": 0.27,
        "height": 0.64,
    },
    # "Republic of Egypt" title + card title, top right.
    "header": {
        "x": 0.42,
        "y": 0.02,
        "width": 0.56,
        "height": 0.22,
    },
    # Full name, two lines, right of the pyramids graphic.
    "name": {
        "x": 0.35,
        "y": 0.25,
        "width": 0.63,
        "height": 0.21,
    },
    # Religion / marital status line.
    "religion_status": {
        "x": 0.35,
        "y": 0.47,
        "width": 0.63,
        "height": 0.11,
    },
    # Address line(s).
    "address": {
        "x": 0.35,
        "y": 0.58,
        "width": 0.63,
        "height": 0.12,
    },
    # National ID number - expanded to full width to ensure no digits are cut off
    "national_id_number": {
        "x": 0.05,
        "y": 0.70,
        "width": 0.90,
        "height": 0.20,
    },
    # Emblem + issue/expiry number, directly under the photo.
    "emblem_area": {
        "x": 0.01,
        "y": 0.70,
        "width": 0.28,
        "height": 0.20,
    },
    # Machine-readable serial number, bottom-left strip.
    "serial_number": {
        "x": 0.00,
        "y": 0.85,
        "width": 0.60,
        "height": 0.15,
    },
}


# ============================================================
# Relative -> pixel conversion
# ============================================================

def _validate_region(
    name,
    x,
    y,
    width,
    height,
    image_width,
    image_height,
):
    """
    Ensure a region's pixel rectangle stays inside the image.
    """

    if x < 0 or y < 0:
        raise ValueError(
            f"Region '{name}' has a negative origin: "
            f"({x}, {y})"
        )

    if width <= 0 or height <= 0:
        raise ValueError(
            f"Region '{name}' has a non-positive size: "
            f"({width}, {height})"
        )

    if x + width > image_width or y + height > image_height:
        raise ValueError(
            f"Region '{name}' exceeds image bounds: "
            f"x+width={x + width}, y+height={y + height}, "
            f"image=({image_width}, {image_height})"
        )


def get_card_regions(
    image: np.ndarray,
):
    """
    Convert CARD_REGIONS relative ratios into pixel coordinates
    for the given (already normalized) card image.

    Returns:
        dict mapping region name -> pixel {"x", "y", "width", "height"}
    """

    image_height, image_width = image.shape[:2]

    regions = {}

    for name, region in CARD_REGIONS.items():

        x = int(region["x"] * image_width)
        y = int(region["y"] * image_height)
        width = int(region["width"] * image_width)
        height = int(region["height"] * image_height)

        _validate_region(
            name,
            x,
            y,
            width,
            height,
            image_width,
            image_height,
        )

        regions[name] = {
            "x": x,
            "y": y,
            "width": width,
            "height": height,
        }

    return regions


# ============================================================
# Debug drawing
# ============================================================

def draw_card_regions_debug(
    image: np.ndarray,
    regions: dict,
):
    """
    Draw all regions with labels on top of the normalized card.
    """

    output = image.copy()

    for name, region in regions.items():

        x = region["x"]
        y = region["y"]
        width = region["width"]
        height = region["height"]

        cv2.rectangle(
            output,
            (x, y),
            (x + width, y + height),
            (0, 255, 0),
            2,
        )

        cv2.putText(
            output,
            name,
            (
                x + 4,
                max(15, y - 6),
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            1,
            cv2.LINE_AA,
        )

    return output


# ============================================================
# Test
# ============================================================

def main():

    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Image not found: "
            f"{INPUT_PATH}"
        )

    image = cv2.imread(
        str(INPUT_PATH)
    )

    if image is None:
        raise RuntimeError(
            "Could not read image."
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Reuse the existing detection -> correction -> normalization
    # pipeline, unmodified.
    # --------------------------------------------------------

    corners = detect_card(
        image
    )

    corrected_card = perspective_correct(
        image,
        corners,
    )

    normalized_card = normalize_card(
        corrected_card
    )

    normalized_height, normalized_width = (
        normalized_card.shape[:2]
    )

    print(
        f"Normalized image: "
        f"{normalized_width} x "
        f"{normalized_height}"
    )

    # --------------------------------------------------------
    # Relative regions
    # --------------------------------------------------------

    regions = get_card_regions(
        normalized_card
    )

    print(
        "\nRegions:"
    )

    for name, region in regions.items():

        print(
            f"{name}: "
            f"x={region['x']}, "
            f"y={region['y']}, "
            f"w={region['width']}, "
            f"h={region['height']}"
        )

    # --------------------------------------------------------
    # Debug output
    # --------------------------------------------------------

    debug = draw_card_regions_debug(
        normalized_card,
        regions,
    )

    debug_path = (
        OUTPUT_DIR
        / "card_regions.jpg"
    )

    cv2.imwrite(
        str(debug_path),
        debug,
    )

    print(
        f"\nDebug image:"
    )

    print(
        debug_path
    )


if __name__ == "__main__":
    main()
