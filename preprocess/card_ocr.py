"""
OCR processing for the normalized National ID card's relative
regions (see preprocess/card_regions.py).

    Region -> Crop -> PaddleOCR -> Raw OCR Result

Also adds a preprocessing-variant layer for difficult fields:

    Region -> Multiple preprocessing variants -> PaddleOCR on
    each -> score each result -> select the best one
"""

import re
import sys

import cv2
import numpy as np

from paddleocr import PaddleOCR

from preprocess.card_detector import (
    INPUT_PATH,
    OUTPUT_DIR,
    detect_card,
    perspective_correct,
    normalize_card,
)

from preprocess.card_regions import get_card_regions

# Windows consoles default to cp1252, which cannot print
# Arabic OCR results.
sys.stdout.reconfigure(encoding="utf-8")


OCR_DEBUG_DIR = OUTPUT_DIR / "ocr"
OCR_VARIANTS_DIR = OUTPUT_DIR / "ocr_variants"

# Text regions sent to PaddleOCR. "photo" and "emblem_area" hold
# images, not text, so they are intentionally excluded.
TEXT_REGIONS = (
    "header",
    "name",
    "religion_status",
    "address",
    "national_id_number",
    "serial_number",
)


_ocr_instance = None


def get_ocr():
    """
    Lazily create a single shared PaddleOCR instance, reusing
    the same configuration as the existing OCR pipeline.

    enable_mkldnn=False works around a PaddlePaddle 3.3.1 +
    oneDNN CPU bug ("ConvertPirAttribute2RuntimeAttribute not
    support ...") that also reproduces on the unmodified
    test_ocr.py in this environment.
    """

    global _ocr_instance

    if _ocr_instance is None:
        _ocr_instance = PaddleOCR(
            lang="ar",
            enable_mkldnn=False,
            use_angle_cls=True,
            show_log=False,
        )

    return _ocr_instance


def crop_region(
    image,
    region,
):
    """
    Crop a single region (in pixel coordinates) out of the
    normalized card image.
    """

    x = region["x"]
    y = region["y"]
    width = region["width"]
    height = region["height"]

    return image[
        y : y + height,
        x : x + width,
    ]


def run_ocr_on_regions(
    normalized_card,
    regions,
    ocr=None,
    save_debug_crops=False,
):
    """
    Run PaddleOCR separately on each text region.

    Returns:
        dict mapping region name -> list of raw OCR entries:
            {"text": str, "confidence": float, "box": [l, t, r, b]}

        Each box is translated back into normalized-card
        coordinates (i.e. relative to the full 1400 x 840 card,
        not the region crop).
    """

    if ocr is None:
        ocr = get_ocr()

    if save_debug_crops:
        OCR_DEBUG_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

    results = {}

    for name in TEXT_REGIONS:

        region = regions.get(name)

        if region is None:
            results[name] = []
            continue

        crop = crop_region(
            normalized_card,
            region,
        )

        if save_debug_crops:
            cv2.imwrite(
                str(OCR_DEBUG_DIR / f"{name}.jpg"),
                crop,
            )

        prediction = ocr.ocr(crop, cls=False)

        entries = []

        for page in prediction:
            if page is None:
                continue

            for line in page:
                box = line[0]
                text, score = line[1]

                # box is [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
                left = min(p[0] for p in box)
                right = max(p[0] for p in box)
                top = min(p[1] for p in box)
                bottom = max(p[1] for p in box)

                # Map the crop-local box back onto the full
                # normalized card so results from every region
                # share one coordinate space.
                entries.append(
                    {
                        "text": text,
                        "confidence": float(score),
                        "box": [
                            float(left + region["x"]),
                            float(top + region["y"]),
                            float(right + region["x"]),
                            float(bottom + region["y"]),
                        ],
                    }
                )

        results[name] = entries

    return results


# ============================================================
# OCR preprocessing variants
# ============================================================
#
# run_ocr_on_regions() above remains available unchanged as a
# simple, single-pass fallback. The functions below add an
# optional layer that tries several preprocessing variants per
# field and automatically picks the best-scoring OCR result.

def _to_bgr(image):
    """
    PaddleOCR is always fed a 3-channel image so every variant
    (color or single-channel) has a consistent shape.
    """

    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    return image


def preprocess_ocr_variants(
    crop,
    field_name=None,
):
    """
    Generate multiple preprocessing variants of a region crop.

    The same variants are generated for every field; only the
    scoring (see score_ocr_result) is field-specific, so the
    original/color image can still win for fields that don't
    benefit from thresholding.

    Returns:
        dict mapping variant name -> BGR image.
    """

    variants = {}

    variants["original"] = crop

    gray = cv2.cvtColor(
        crop,
        cv2.COLOR_BGR2GRAY,
    )

    variants["grayscale"] = _to_bgr(gray)

    upscale_2x = cv2.resize(
        gray,
        None,
        fx=2.0,
        fy=2.0,
        interpolation=cv2.INTER_CUBIC,
    )

    variants["upscale_2x"] = _to_bgr(
        upscale_2x
    )

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8),
    )

    variants["clahe"] = _to_bgr(
        clahe.apply(gray)
    )

    variants["upscale_clahe"] = _to_bgr(
        clahe.apply(upscale_2x)
    )

    # Odd block size scaled to the crop, so small regions
    # (e.g. serial_number) don't get an oversized neighborhood.
    block_size = max(
        3,
        (min(gray.shape[:2]) // 3) | 1,
    )

    variants["adaptive_threshold"] = _to_bgr(
        cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            block_size,
            10,
        )
    )

    _, otsu = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )

    variants["otsu_threshold"] = _to_bgr(
        otsu
    )

    # NEW: Dilation variant specifically for tiny dots (zeros) and thin fonts
    kernel = np.ones((2, 2), np.uint8)
    dilated = cv2.erode(otsu, kernel, iterations=1) # Erosion on binary inverse = Dilation on text
    variants["thickened"] = _to_bgr(dilated)

    upscale_3x = cv2.resize(
        gray,
        None,
        fx=3.0,
        fy=3.0,
        interpolation=cv2.INTER_CUBIC,
    )

    variants["upscale_3x_contrast"] = _to_bgr(
        cv2.equalizeHist(upscale_3x)
    )

    return variants


def _run_ocr_on_image(
    ocr,
    image,
):
    """
    Run PaddleOCR on a single image and return its raw entries.
    """

    prediction = ocr.ocr(image, cls=False)

    entries = []

    for page in prediction:
        if page is None:
            continue

        for line in page:
            box = line[0]
            text, score = line[1]

            # Convert polygon [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
            # to [left, top, right, bottom] for compatibility with
            # _order_entries_for_reading
            left = min(p[0] for p in box)
            right = max(p[0] for p in box)
            top = min(p[1] for p in box)
            bottom = max(p[1] for p in box)

            entries.append(
                {
                    "text": text,
                    "confidence": float(score),
                    "box": [left, top, right, bottom],
                }
            )

    return entries


def _order_entries_for_reading(
    entries,
):
    """
    Order OCR entries into natural reading order: top-to-bottom
    lines, right-to-left within each line (Arabic script). Raw
    detection order is often left-to-right regardless of
    script, which silently reverses joined Arabic phrases.
    """

    if not entries:
        return entries

    def _center_y(entry):
        top, bottom = entry["box"][1], entry["box"][3]
        return (top + bottom) / 2.0

    def _center_x(entry):
        left, right = entry["box"][0], entry["box"][2]
        return (left + right) / 2.0

    def _height(entry):
        return entry["box"][3] - entry["box"][1]

    lines = []

    for entry in sorted(entries, key=_center_y):

        placed = False

        for line in lines:
            reference = line[0]

            if abs(_center_y(entry) - _center_y(reference)) <= (
                max(_height(entry), _height(reference)) * 0.6
            ):
                line.append(entry)
                placed = True
                break

        if not placed:
            lines.append([entry])

    ordered = []

    for line in lines:
        ordered.extend(
            sorted(line, key=_center_x, reverse=True)
        )

    return ordered


def _summarize_entries(
    entries,
):
    """
    Collapse a variant's per-box OCR entries into one text
    string and one average confidence value.
    """

    ordered_entries = _order_entries_for_reading(
        entries
    )

    texts = [
        entry["text"]
        for entry in ordered_entries
    ]

    confidences = [
        entry["confidence"]
        for entry in entries
    ]

    text = " ".join(texts)

    confidence = (
        sum(confidences) / len(confidences)
        if confidences
        else 0.0
    )

    return {
        "text": text,
        "confidence": confidence,
        "entries": entries,
    }


def run_ocr_variants(
    crop,
    field_name,
    ocr=None,
):
    """
    Generate preprocessing variants for a region and run
    PaddleOCR on every one of them.

    Returns:
        dict mapping variant name ->
            {"text": str, "confidence": float, "entries": [...]}
    """

    if ocr is None:
        ocr = get_ocr()

    variants = preprocess_ocr_variants(
        crop,
        field_name,
    )

    return {
        variant_name: _summarize_entries(
            _run_ocr_on_image(ocr, variant_image)
        )
        for variant_name, variant_image in variants.items()
    }


# ============================================================
# Scoring
# ============================================================

_ARABIC_RE = re.compile(r"[\u0600-\u06FF]")
_DIGIT_RE = re.compile(r"[0-9\u0660-\u0669\u06F0-\u06F9]")
_ALNUM_RE = re.compile(r"[A-Za-z0-9\u0660-\u0669\u06F0-\u06F9]")
_VALID_CHAR_RE = re.compile(r"[A-Za-z0-9\u0600-\u06FF\s]")

# Egyptian National ID numbers are 14 digits long.
EXPECTED_NATIONAL_ID_DIGITS = 14

ARABIC_INDIC_DIGITS = "٠١٢٣٤٥٦٧٨٩"
EXTENDED_ARABIC_INDIC_DIGITS = "۰۱۲۳۴۵۶۷۸۹"  # U+06F0..U+06F9 (Persian/Urdu)
ENGLISH_DIGITS = "0123456789"
DIGIT_MAP = str.maketrans(
    ARABIC_INDIC_DIGITS + EXTENDED_ARABIC_INDIC_DIGITS,
    ENGLISH_DIGITS + ENGLISH_DIGITS,
)

# Characters that Egyptian ID zero-groups (printed as small dots) can be
# recognized as. All are treated as "0". (Only true dot/middot glyphs --
# never spaces or letters, which would inject false zeros.)
ZERO_LIKE_CHARS = ".\u00b7\u2022\u2024\u2027\u2219\u22c5\u06d4\u066b\u066c\u2025\u2026"

def normalize_digits(text):
    """Convert Arabic-Indic digits to English digits and handle zeros."""
    if not text: return ""
    # The zero groups on Egyptian IDs are printed as small dots, which OCR
    # often returns as one of several dot/middot glyphs. Treat them as zeros.
    for ch in ZERO_LIKE_CHARS:
        text = text.replace(ch, "0")
    return text.translate(DIGIT_MAP)


def _is_plausible_egyptian_id(digits):
    """
    Check whether a 14-digit string is a structurally valid Egyptian
    National ID. Used to decide the correct reading orientation, since
    the Arabic OCR model emits digits right-to-left (reversed).

    Layout: C YY MM DD GG SSS G
        C  century (2 = 1900s, 3 = 2000s)
        YY year, MM month (01-12), DD day (01-31)
        GG governorate code, SSSS serial, last = checksum/gender
    """
    if len(digits) != 14 or not digits.isdigit():
        return False
    if digits[0] not in ("2", "3"):
        return False
    month = int(digits[3:5])
    day = int(digits[5:7])
    if not (1 <= month <= 12):
        return False
    if not (1 <= day <= 31):
        return False
    if int(digits[7:9]) == 0:
        return False
    return True


def orient_national_id(digits):
    """
    Return the National ID digits in correct left-to-right order.

    The Arabic OCR recognizer reads numbers right-to-left, which flips
    the digit order (e.g. 30007263400037 -> 73000436270003). We pick
    whichever orientation forms a structurally valid Egyptian ID; if
    neither validates (e.g. some digits were dropped), fall back to the
    rule that Egyptian IDs always start with 2 or 3.
    """
    if not digits:
        return digits

    reversed_digits = digits[::-1]

    if _is_plausible_egyptian_id(digits):
        return digits
    if _is_plausible_egyptian_id(reversed_digits):
        return reversed_digits

    # Partial capture fallback: an ID starts with 2/3, never ends with it.
    if digits[0] not in ("2", "3") and digits[-1] in ("2", "3"):
        return reversed_digits

    return digits

def clean_arabic_text(text):
    """Fix character reversal for Arabic names, but leave digits alone."""
    if not text: return ""
    
    # If it's mostly digits, don't reverse it (numbers are L-to-R)
    if sum(c.isdigit() or c in ARABIC_INDIC_DIGITS or c == "." for c in text) > len(text) / 3:
        return normalize_digits(text)

    # Reverse characters in each word for Arabic text
    words = text.split()
    fixed_words = [w[::-1] for w in words]
    text = " ".join(fixed_words)
    
    # Keep only Arabic characters and spaces
    text = re.sub(r"[^\u0600-\u06FF\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def extract_clean_data(selected_results, original_image_id=None):
    """
    Final extraction layer. Merges religion and address, and improves ID capture.
    """
    cleaned = {}

    # 1. Full Name
    name_raw = selected_results.get("name", {}).get("text", "")
    cleaned["full_name"] = clean_arabic_text(name_raw)

    # 2. National ID Number
    # Priority 1: Use the brute-force search result from the original image
    # Priority 2: Use the cropped region result
    id_raw = original_image_id or selected_results.get("national_id_number", {}).get("text", "")

    id_norm = normalize_digits(id_raw)
    id_digits = "".join(re.findall(r"[0-9]", id_norm))

    # Arabic OCR reads numbers right-to-left, flipping the digit order.
    # Restore correct left-to-right order (e.g. 73000436270003 -> 30007263400037).
    id_digits = orient_national_id(id_digits)

    cleaned["national_id"] = id_digits
    cleaned["id_is_valid"] = len(id_digits) == EXPECTED_NATIONAL_ID_DIGITS

    # 3. Merged Address (Religion/Status + Address)
    rel_raw = selected_results.get("religion_status", {}).get("text", "")
    addr_raw = selected_results.get("address", {}).get("text", "")

    full_address = f"{clean_arabic_text(rel_raw)} {clean_arabic_text(addr_raw)}".strip()
    cleaned["address"] = full_address

    return cleaned

def _char_ratio(text, pattern):
    if not text:
        return 0.0

    matches = len(pattern.findall(text))

    return matches / len(text)


def score_ocr_result(
    result,
    field_name,
):
    """
    Score a single OCR result (one preprocessing variant) for
    how likely it is to be a good read of `field_name`.

    Only general, field-agnostic characteristics are used
    (confidence, text length, digit/Arabic/alnum ratios,
    number of detected boxes) -- never the field's actual
    expected value.
    """

    text = (result.get("text") or "").strip()
    confidence = result.get("confidence", 0.0) or 0.0
    entries = result.get("entries") or []

    # Never let an empty result outscore a non-empty one.
    if not text:
        return 0.0

    length_score = min(
        len(text) / 20.0,
        1.0,
    )

    # Capped at 5 boxes/words so a fuller, multi-line result
    # (e.g. a 2-line name) isn't beaten by a shorter but very
    # slightly more confident partial read.
    box_score = min(
        len(entries) / 5.0,
        1.0,
    )

    valid_ratio = _char_ratio(
        text,
        _VALID_CHAR_RE,
    )

    digit_ratio = _char_ratio(
        text,
        _DIGIT_RE,
    )

    arabic_ratio = _char_ratio(
        text,
        _ARABIC_RE,
    )

    alnum_ratio = _char_ratio(
        text,
        _ALNUM_RE,
    )

    score = (
        confidence * 0.30
        + length_score * 0.15
        + box_score * 0.20
        + valid_ratio * 0.15
    )

    if field_name == "national_id_number":

        digit_count = sum(
            1
            for character in text
            if _DIGIT_RE.match(character)
        )

        length_closeness = 1.0 - min(
            abs(digit_count - EXPECTED_NATIONAL_ID_DIGITS)
            / EXPECTED_NATIONAL_ID_DIGITS,
            1.0,
        )

        score += digit_ratio * 0.50
        score += length_closeness * 0.20

    elif field_name == "serial_number":

        score += alnum_ratio * 0.45

    elif field_name == "religion_status":

        score += arabic_ratio * 0.45

    else:

        score += arabic_ratio * 0.20

    return max(0.0, score)


def select_best_result(
    results,
    field_name,
):
    """
    Score every variant's OCR result and return the winner.

    Returns:
        {
            "field": str,
            "text": str,
            "confidence": float,
            "selected_variant": str,
        }
    """

    if not results:
        return {
            "field": field_name,
            "text": "",
            "confidence": 0.0,
            "selected_variant": None,
        }

    best_variant = max(
        results,
        key=lambda variant_name: score_ocr_result(
            results[variant_name],
            field_name,
        ),
    )

    best = results[best_variant]

    return {
        "field": field_name,
        "text": best["text"],
        "confidence": best["confidence"],
        "selected_variant": best_variant,
    }


# ============================================================
# Test
# ============================================================

def process_image(image_path, ocr=None):
    """
    Complete pipeline: Detection -> Normalization -> OCR -> Extraction.
    Returns the final cleaned data dictionary.
    """
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Could not read image at {image_path}")

    if ocr is None:
        ocr = get_ocr()

    # 1. Pipeline orchestration
    from preprocess.card_regions import get_card_regions
    
    # BRUTE FORCE SEARCH for the 14-digit ID in the whole image
    # This is often more reliable than crops because context helps the OCR.
    raw_full_ocr = _run_ocr_on_image(ocr, image)
    best_full_id = None
    
    # Join all detected numeric fragments in the bottom-right of the image
    # (where the National ID sits) into one string.
    bottom_digits = []
    img_h, img_w = image.shape[0], image.shape[1]
    for entry in raw_full_ocr:
        box = entry["box"]
        center_y = (box[1] + box[3]) / 2
        center_x = (box[0] + box[2]) / 2
        # The ID number occupies the bottom band and the right ~60% of the
        # card. Skipping the left side avoids the date stamp / serial number.
        if center_y > img_h * 0.6 and center_x > img_w * 0.35:
            digits = "".join(re.findall(r"[0-9]", normalize_digits(entry["text"])))
            if digits:
                bottom_digits.append((box[0], digits)) # (x_coord, digits)

    # The recognizer reverses each fragment's digits, so join the fragments
    # right-to-left (Arabic reading order). orient_national_id() then flips
    # the whole string back to correct left-to-right order.
    bottom_digits.sort(key=lambda x: x[0], reverse=True)
    combined_digits = "".join([d[1] for d in bottom_digits])

    if len(combined_digits) >= 10:
        best_full_id = combined_digits

    corners = detect_card(image)
    corrected_card = perspective_correct(image, corners)
    normalized_card = normalize_card(corrected_card)
    
    # Using corrected_card (High Res) for better OCR results on small digits
    high_res_regions = get_card_regions(corrected_card)

    selected_results = {}

    for name in TEXT_REGIONS:
        region = high_res_regions[name]
        crop = crop_region(corrected_card, region)

        # Run variants and pick best
        variant_results = run_ocr_variants(crop, name, ocr=ocr)
        best = select_best_result(variant_results, name)
        selected_results[name] = best

    # 2. Final Extraction with original image fallback for the ID
    return extract_clean_data(selected_results, original_image_id=best_full_id)

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
    # -> region pipeline, unmodified.
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

    regions = get_card_regions(
        normalized_card
    )

    # --------------------------------------------------------
    # OCR
    # --------------------------------------------------------

    print(
        "Initializing PaddleOCR..."
    )

    ocr = get_ocr()

    print(
        "PaddleOCR initialized!"
    )

    results = run_ocr_on_regions(
        normalized_card,
        regions,
        ocr=ocr,
        save_debug_crops=True,
    )

    # Write a plain UTF-8 report in addition to the console
    # output, since Windows terminals often mangle Arabic text.
    report_lines = []

    for name in TEXT_REGIONS:

        print(
            f"\n=== {name.upper()} ==="
        )

        report_lines.append(
            f"\n=== {name.upper()} ==="
        )

        entries = results.get(name, [])

        if not entries:
            print(
                "(no text detected)"
            )
            report_lines.append(
                "(no text detected)"
            )
            continue

        for entry in entries:
            line = (
                f"{entry['text']} "
                f"(confidence="
                f"{entry['confidence']:.2f})"
            )
            print(line)
            report_lines.append(line)

    report_path = (
        OUTPUT_DIR
        / "ocr_results.txt"
    )

    report_path.write_text(
        "\n".join(report_lines),
        encoding="utf-8",
    )

    print(
        f"\nOCR debug crops:"
    )

    print(
        OCR_DEBUG_DIR
    )

    # --------------------------------------------------------
    # OCR with preprocessing variants + automatic selection
    # --------------------------------------------------------
    #
    # Cropped from corrected_card (pre-normalization, full
    # resolution) instead of the fixed 1400x840 normalized_card:
    # small dense text (national_id_number, serial_number) gets
    # blurred away by that downscale. get_card_regions() is
    # reused unchanged -- it already computes ratios relative to
    # whatever image it's given.

    high_res_regions = get_card_regions(
        corrected_card
    )

    OCR_VARIANTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    variant_report_lines = []

    selected_results = {}

    for name in TEXT_REGIONS:

        region = high_res_regions[name]

        crop = crop_region(
            corrected_card,
            region,
        )

        variants = preprocess_ocr_variants(
            crop,
            name,
        )

        field_dir = (
            OCR_VARIANTS_DIR
            / name
        )

        field_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        for variant_name, variant_image in variants.items():
            cv2.imwrite(
                str(field_dir / f"{variant_name}.jpg"),
                variant_image,
            )

        variant_results = {
            variant_name: _summarize_entries(
                _run_ocr_on_image(ocr, variant_image)
            )
            for variant_name, variant_image in variants.items()
        }

        best = select_best_result(
            variant_results,
            name,
        )

        selected_results[name] = best

        header_line = (
            f"\n=== {name.upper()} ==="
        )

        print(header_line)
        variant_report_lines.append(header_line)

        for variant_name, result in variant_results.items():

            score = score_ocr_result(
                result,
                name,
            )

            block = (
                f"\n{variant_name}:\n"
                f'  text="{result["text"]}"\n'
                f'  confidence={result["confidence"]:.2f}\n'
                f"  score={score:.2f}"
            )

            print(block)
            variant_report_lines.append(block)

        selected_block = (
            f"\nSELECTED:\n"
            f"  variant={best['selected_variant']}\n"
            f'  text="{best["text"]}"\n'
            f"  confidence={best['confidence']:.2f}"
        )

        print(selected_block)
        variant_report_lines.append(selected_block)

    variant_report_path = (
        OUTPUT_DIR
        / "ocr_variant_results.txt"
    )

    variant_report_path.write_text(
        "\n".join(variant_report_lines),
        encoding="utf-8",
    )

    print(
        f"\nOCR variant crops:"
    )

    print(
        OCR_VARIANTS_DIR
    )

    print(
        f"\nOCR variant report:"
    )

    print(
        variant_report_path
    )

    print(
        f"\nSelected results summary:"
    )

    for name in TEXT_REGIONS:
        print(
            selected_results[name]
        )

    # --------------------------------------------------------
    # FINAL CLEAN EXTRACTION
    # --------------------------------------------------------
    print("\n" + "="*50)
    print("FINAL EXTRACTED DATA")
    print("="*50)
    
    final_data = extract_clean_data(selected_results)
    
    import json
    print(json.dumps(final_data, indent=2, ensure_ascii=False))

    # Save to JSON
    json_path = OUTPUT_DIR / "final_data.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(final_data, f, indent=2, ensure_ascii=False)
        
    print(f"\nFinal data saved to: {json_path}")


if __name__ == "__main__":
    main()
