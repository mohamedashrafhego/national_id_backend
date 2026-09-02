from pathlib import Path

import cv2
import numpy as np


INPUT_PATH = Path("national_id.jpg")
OUTPUT_DIR = Path("debug_card_detection")


def resize_for_debug(
    image: np.ndarray,
    max_side: int = 1000,
):
    height, width = image.shape[:2]

    scale = min(
        1.0,
        max_side / max(height, width),
    )

    if scale == 1.0:
        return image.copy()

    return cv2.resize(
        image,
        None,
        fx=scale,
        fy=scale,
        interpolation=cv2.INTER_AREA,
    )


def save_gray(
    path: Path,
    image: np.ndarray,
):
    cv2.imwrite(
        str(path),
        image,
    )


def create_color_difference(
    image: np.ndarray,
) -> np.ndarray:
    """
    Estimate how different each pixel is from the
    image background.

    Background color is estimated dynamically from
    the image borders.

    No fixed card coordinates are used.
    """

    height, width = image.shape[:2]

    # Take pixels from the outer border of the image.
    border_ratio = 0.04

    bx = max(
        1,
        int(width * border_ratio),
    )

    by = max(
        1,
        int(height * border_ratio),
    )

    border_pixels = np.concatenate(
        [
            image[:by, :, :].reshape(-1, 3),
            image[-by:, :, :].reshape(-1, 3),
            image[:, :bx, :].reshape(-1, 3),
            image[:, -bx:, :].reshape(-1, 3),
        ],
        axis=0,
    )

    # Median is more robust than mean against shadows/noise.
    background_color = np.median(
        border_pixels,
        axis=0,
    ).astype(np.float32)

    image_float = image.astype(
        np.float32
    )

    difference = np.sqrt(
        np.sum(
            (
                image_float
                - background_color
            )
            ** 2,
            axis=2,
        )
    )

    # Normalize to 0..255.
    difference = cv2.normalize(
        difference,
        None,
        0,
        255,
        cv2.NORM_MINMAX,
    )

    return difference.astype(
        np.uint8
    )


def create_foreground_mask(
    difference: np.ndarray,
) -> np.ndarray:
    """
    Convert color difference into a foreground mask.

    Otsu chooses the threshold dynamically.
    """

    # Slight blur removes tiny pixel-level variations.
    blurred = cv2.GaussianBlur(
        difference,
        (9, 9),
        0,
    )

    _, mask = cv2.threshold(
        blurred,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )

    height, width = mask.shape

    # Morphology size is relative to image dimensions.
    kernel_size = max(
        3,
        int(
            round(
                min(height, width)
                * 0.015
            )
        )
        | 1,
    )

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (
            kernel_size,
            kernel_size,
        ),
    )

    # Connect regions belonging to the card.
    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel,
    )

    # Remove small isolated objects.
    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel,
    )

    return mask


def create_contour_debug(
    image: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:

    output = image.copy()

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    height, width = mask.shape

    image_area = height * width

    print(
        f"\nContours found: {len(contours)}"
    )

    candidates = []

    for index, contour in enumerate(
        contours
    ):

        area = cv2.contourArea(
            contour
        )

        if area <= 0:
            continue

        x, y, w, h = cv2.boundingRect(
            contour
        )

        rect_area = w * h

        if rect_area <= 0:
            continue

        rectangularity = (
            area / rect_area
        )

        ratio = max(w, h) / min(w, h)

        area_ratio = (
            area / image_area
        )

        print(
            f"Contour {index}: "
            f"area={area:.0f}, "
            f"area_ratio={area_ratio:.3f}, "
            f"bbox=({x},{y},{w},{h}), "
            f"ratio={ratio:.3f}, "
            f"rectangularity={rectangularity:.3f}"
        )

        # Candidate filtering is intentionally
        # broad at this diagnostic stage.
        if area_ratio < 0.10:
            continue

        if not 1.2 <= ratio <= 2.0:
            continue

        if rectangularity < 0.55:
            continue

        candidates.append(
            (
                area,
                contour,
            )
        )

    candidates.sort(
        key=lambda item: item[0],
        reverse=True,
    )

    print(
        f"\nCard-like candidates: "
        f"{len(candidates)}"
    )

    # Draw all large candidates.
    for rank, (
        area,
        contour,
    ) in enumerate(
        candidates[:10],
        start=1,
    ):

        color = (
            0,
            255,
            0,
        )

        cv2.drawContours(
            output,
            [contour],
            -1,
            color,
            4,
        )

        x, y, w, h = cv2.boundingRect(
            contour
        )

        cv2.putText(
            output,
            f"Candidate {rank}",
            (
                x,
                max(
                    30,
                    y - 10,
                ),
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (
                0,
                255,
                0,
            ),
            3,
        )

    return output


def create_gray(
    image: np.ndarray,
) -> np.ndarray:

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY,
    )

    return gray


def main():

    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Image not found: {INPUT_PATH}"
        )

    image = cv2.imread(
        str(INPUT_PATH)
    )

    if image is None:
        raise RuntimeError(
            f"Could not read image: {INPUT_PATH}"
        )

    print(
        f"Original image size: "
        f"{image.shape[1]} x "
        f"{image.shape[0]}"
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Detection/debug image.
    working_image = resize_for_debug(
        image
    )

    print(
        f"Debug image size: "
        f"{working_image.shape[1]} x "
        f"{working_image.shape[0]}"
    )

    # -------------------------------------------------
    # 01 - GRAYSCALE
    # -------------------------------------------------

    gray = create_gray(
        working_image
    )

    save_gray(
        OUTPUT_DIR / "01_gray.jpg",
        gray,
    )

    # -------------------------------------------------
    # 02 - COLOR DIFFERENCE
    # -------------------------------------------------

    difference = create_color_difference(
        working_image
    )

    save_gray(
        OUTPUT_DIR
        / "02_background_difference.jpg",
        difference,
    )

    # -------------------------------------------------
    # 03 - FOREGROUND MASK
    # -------------------------------------------------

    foreground_mask = create_foreground_mask(
        difference
    )

    save_gray(
        OUTPUT_DIR
        / "03_foreground_mask.jpg",
        foreground_mask,
    )

    # -------------------------------------------------
    # 04 - CONTOURS
    # -------------------------------------------------

    contour_debug = create_contour_debug(
        working_image,
        foreground_mask,
    )

    cv2.imwrite(
        str(
            OUTPUT_DIR
            / "04_contours.jpg"
        ),
        contour_debug,
    )

    print(
        "\nDebug files generated:"
    )

    print(
        OUTPUT_DIR
        / "01_gray.jpg"
    )

    print(
        OUTPUT_DIR
        / "02_background_difference.jpg"
    )

    print(
        OUTPUT_DIR
        / "03_foreground_mask.jpg"
    )

    print(
        OUTPUT_DIR
        / "04_contours.jpg"
    )


if __name__ == "__main__":
    main()