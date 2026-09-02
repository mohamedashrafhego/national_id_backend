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

        prediction = ocr.predict(crop)

        entries = []

        for page in prediction:

            texts = page.get("rec_texts", [])
            scores = page.get("rec_scores", [])
            boxes = page.get("rec_boxes", [])

            for text, score, box in zip(texts, scores, boxes):

                left, top, right, bottom = box

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

    prediction = ocr.predict(image)

    entries = []

    for page in prediction:

        texts = page.get("rec_texts", [])
        scores = page.get("rec_scores", [])
        boxes = page.get("rec_boxes", [])

        for text, score, box in zip(texts, scores, boxes):
            entries.append(
                {
                    "text": text,
                    "confidence": float(score),
                    "box": box,
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


if __name__ == "__main__":
    main()
