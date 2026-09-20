
import re

from PIL import Image
import os

# --------------------------------------------------
# Arabic digits → English digits
# --------------------------------------------------

ARABIC_DIGITS = str.maketrans(
    "٠١٢٣٤٥٦٧٨٩",
    "0123456789"
)


def normalize_digits(text: str) -> str:
    """
    Convert Arabic-Indic digits to English digits.

    Example:
    ٣٠٠٠٧٢٦٣٤٠٠٠٣٧
    →
    30007263400037
    """

    return text.translate(ARABIC_DIGITS)


# --------------------------------------------------
# Text helpers
# --------------------------------------------------

def is_arabic_text(text: str) -> bool:
    """
    Check if text contains Arabic characters.
    """

    return bool(
        re.search(
            r"[\u0600-\u06FF]",
            text
        )
    )


def clean_text(text: str) -> str:
    """
    Clean OCR text.
    """

    if not text:
        return ""

    text = text.strip()

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text


# --------------------------------------------------
# National ID
# --------------------------------------------------

def extract_national_id(items):
    """
    Extract Egyptian National ID.

    Expected format:
    14 digits

    Supports:
    - English digits
    - Arabic-Indic digits
    """

    candidates = []

    for item in items:

        text = normalize_digits(
            item.text
        )

        digits = re.sub(
            r"\D",
            "",
            text
        )

        if len(digits) == 14:

            candidates.append(item)

    if not candidates:
        return None

    # Highest confidence first
    candidates.sort(
        key=lambda item: item.confidence,
        reverse=True
    )

    item = candidates[0]

    value = re.sub(
        r"\D",
        "",
        normalize_digits(item.text)
    )

    return {
        "value": value,
        "confidence": item.confidence,
    }


# --------------------------------------------------
# Name
# --------------------------------------------------

def extract_name(items):
    """
    Extract the full name from the Egyptian ID front.

    Current test layout:
    Name is around Y = 1278.
    """

    candidates = []

    for item in items:

        text = clean_text(
            item.text
        )

        if not text:
            continue

        if not is_arabic_text(text):
            continue

        if item.confidence < 0.75:
            continue

        # Current image name area
        if 1150 <= item.center_y <= 1400:

            candidates.append(item)

    if not candidates:
        return None

    candidates.sort(
        key=lambda item: item.confidence,
        reverse=True
    )

    item = candidates[0]

    return {
        "value": item.text,
        "confidence": item.confidence,
    }


# --------------------------------------------------
# Address
# --------------------------------------------------

def extract_address(items):
    """
    Extract address from the Egyptian ID front.

    Current test layout:
    Address is around Y = 1450 - 1900.
    """

    candidates = []

    for item in items:

        text = clean_text(
            item.text
        )

        if not text:
            continue

        if not is_arabic_text(text):
            continue

        if item.confidence < 0.75:
            continue

        # Current image address area
        if 1400 <= item.center_y <= 1900:

            candidates.append(item)

    if not candidates:
        return None

    # Arabic is RTL
    candidates.sort(
        key=lambda item: (
            item.center_y,
            -item.center_x
        )
    )

    address = " ".join(
        item.text
        for item in candidates
    )

    confidence = (
        sum(
            item.confidence
            for item in candidates
        )
        / len(candidates)
    )

    return {
        "value": address,
        "confidence": confidence,
    }


# --------------------------------------------------
# Serial Number
# --------------------------------------------------

def extract_serial_number(items):
    """
    Extract the card serial number.

    Example:
    KV6753093
    """

    candidates = []

    for item in items:

        text = clean_text(
            item.text
        )

        if not text:
            continue

        # Example:
        # KV6753093
        if re.fullmatch(
            r"[A-Za-z]{2}\d{7}",
            text
        ):

            candidates.append(item)

    if not candidates:
        return None

    candidates.sort(
        key=lambda item: item.confidence,
        reverse=True
    )

    item = candidates[0]

    return {
        "value": item.text,
        "confidence": item.confidence,
    }

from PIL import Image
import os


def extract_full_name_from_crop(ocr, image_path):
    """
    Extract the full name from the front side of an Egyptian National ID.

    Expected layout:

        محمد
        اشرف محمد علي الحجاوى
    """

    # --------------------------------------------------
    # Load image
    # --------------------------------------------------

    image = Image.open(image_path)

    # --------------------------------------------------
    # Crop name area
    # --------------------------------------------------

    name_crop = image.crop(
        (
            1500,
            1000,
            3350,
            1500,
        )
    )

    # --------------------------------------------------
    # Resize x2
    # --------------------------------------------------

    width, height = name_crop.size

    name_crop = name_crop.resize(
        (width * 2, height * 2),
        Image.Resampling.LANCZOS,
    )

    # --------------------------------------------------
    # Save temp image
    # --------------------------------------------------

    temp_path = "temp_name_crop.jpg"
    name_crop.save(temp_path)

    # --------------------------------------------------
    # OCR
    # --------------------------------------------------

    result = ocr.ocr(temp_path, cls=False)

    items = []

    for page in result:
        if page is None:
            continue

        for line in page:
            box = line[0]
            text, score = line[1]

            text = clean_text(text)

            if not text:
                continue

            if score < 0.70:
                continue

            # box is [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
            left = min(p[0] for p in box)
            right = max(p[0] for p in box)
            top = min(p[1] for p in box)
            bottom = max(p[1] for p in box)

            center_x = (left + right) // 2
            center_y = (top + bottom) // 2

            items.append(
                {
                    "text": text,
                    "confidence": float(score),
                    "center_x": center_x,
                    "center_y": center_y,
                }
            )

    if os.path.exists(temp_path):
        os.remove(temp_path)

    if not items:
        return None

    # --------------------------------------------------
    # Sort by line then RTL
    # --------------------------------------------------

    items.sort(
        key=lambda item: (
            item["center_y"],
            -item["center_x"],
        )
    )

    # --------------------------------------------------
    # Group into lines
    # --------------------------------------------------

    lines = []

    tolerance = 80

    for item in items:

        added = False

        for line in lines:

            if abs(line["y"] - item["center_y"]) < tolerance:

                line["items"].append(item)

                added = True
                break

        if not added:

            lines.append(
                {
                    "y": item["center_y"],
                    "items": [item],
                }
            )

    # --------------------------------------------------
    # Sort each line RTL
    # --------------------------------------------------

    full_name_parts = []
    confidences = []

    for line in sorted(lines, key=lambda x: x["y"]):

        line["items"].sort(
            key=lambda x: -x["center_x"]
        )

        text = " ".join(
            item["text"]
            for item in line["items"]
        )

        full_name_parts.append(text)

        confidences.extend(
            item["confidence"]
            for item in line["items"]
        )

    full_name = " ".join(full_name_parts)

    return {
        "value": full_name,
        "confidence": sum(confidences) / len(confidences),
        "parts": full_name_parts,
    }


    

