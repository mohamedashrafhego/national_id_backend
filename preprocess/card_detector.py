from pathlib import Path

import cv2
import numpy as np


# Egyptian ID cards are approximately 1.59:1
# We use a range instead of one exact value.
MIN_CARD_RATIO = 1.45
MAX_CARD_RATIO = 1.72

# Detection is performed on a smaller image for speed.
MAX_DETECTION_SIDE = 700

# Large dark connected regions are treated as obstacles
# for the segmentation step.
DARK_THRESHOLD = 25

# Minimum size of a dark region relative to the image.
MIN_DARK_AREA_RATIO = 0.005


def resize_for_detection(
    image: np.ndarray,
    max_side: int = MAX_DETECTION_SIDE,
):
    """
    Resize image only for card detection.

    Returns:
        resized_image
        scale
    """

    height, width = image.shape[:2]

    scale = min(
        1.0,
        max_side / max(height, width),
    )

    if scale == 1.0:
        return image.copy(), scale

    resized = cv2.resize(
        image,
        None,
        fx=scale,
        fy=scale,
        interpolation=cv2.INTER_AREA,
    )

    return resized, scale


def detect_large_dark_regions(
    image: np.ndarray,
) -> np.ndarray:
    """
    Detect very large dark connected components.

    This prevents large dark objects / redaction blocks
    from dominating GrabCut and contour detection.

    Returns:
        binary mask
    """

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY,
    )

    dark = np.where(
        gray < DARK_THRESHOLD,
        255,
        0,
    ).astype(np.uint8)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        dark,
        connectivity=8,
    )

    mask = np.zeros_like(dark)

    image_area = image.shape[0] * image.shape[1]
    min_area = image_area * MIN_DARK_AREA_RATIO

    for label in range(1, num_labels):

        area = stats[
            label,
            cv2.CC_STAT_AREA,
        ]

        if area >= min_area:
            mask[labels == label] = 255

    return mask


def remove_dark_regions(
    image: np.ndarray,
    dark_mask: np.ndarray,
) -> np.ndarray:
    """
    Inpaint large dark regions so they don't create
    dominant segmentation boundaries.
    """

    if not np.any(dark_mask):
        return image.copy()

    return cv2.inpaint(
        image,
        dark_mask,
        3,
        cv2.INPAINT_TELEA,
    )


def create_grabcut_mask(
    image: np.ndarray,
    dark_mask: np.ndarray,
) -> np.ndarray:
    """
    Create a relative GrabCut initialization mask.

    No fixed pixel coordinates are used.
    """

    height, width = image.shape[:2]

    mask = np.full(
        (height, width),
        cv2.GC_BGD,
        dtype=np.uint8,
    )

    # Large probable foreground area.
    margin_x = int(width * 0.06)
    margin_y = int(height * 0.08)

    mask[
        margin_y:height - margin_y,
        margin_x:width - margin_x,
    ] = cv2.GC_PR_FGD

    # Smaller sure-foreground region.
    # This is relative to the image dimensions.
    sure_x = int(width * 0.18)
    sure_y = int(height * 0.20)

    mask[
        sure_y:height - sure_y,
        sure_x:width - sure_x,
    ] = cv2.GC_FGD

    # Large dark regions should not become strong foreground seeds.
    mask[dark_mask > 0] = cv2.GC_PR_BGD

    return mask


def grabcut_foreground(
    image: np.ndarray,
    initial_mask: np.ndarray,
) -> np.ndarray:
    """
    Run GrabCut and return a binary foreground mask.
    """

    mask = initial_mask.copy()

    background_model = np.zeros(
        (1, 65),
        dtype=np.float64,
    )

    foreground_model = np.zeros(
        (1, 65),
        dtype=np.float64,
    )

    cv2.grabCut(
        image,
        mask,
        None,
        background_model,
        foreground_model,
        2,
        cv2.GC_INIT_WITH_MASK,
    )

    foreground = np.where(
        (mask == cv2.GC_FGD)
        | (mask == cv2.GC_PR_FGD),
        255,
        0,
    ).astype(np.uint8)

    return foreground


def clean_foreground_mask(
    foreground: np.ndarray,
) -> np.ndarray:
    """
    Remove small gaps and connect the card region.
    """

    height, width = foreground.shape[:2]

    kernel_size = max(
        3,
        int(round(0.015 * min(height, width))) | 1,
    )

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (kernel_size, kernel_size),
    )

    cleaned = cv2.morphologyEx(
        foreground,
        cv2.MORPH_CLOSE,
        kernel,
    )

    return cleaned


def order_points(
    points: np.ndarray,
) -> np.ndarray:
    """
    Return points in this order:

        top-left
        top-right
        bottom-right
        bottom-left
    """

    points = points.astype(np.float32)

    sums = points.sum(axis=1)
    differences = points[:, 1] - points[:, 0]

    top_left = points[np.argmin(sums)]
    top_right = points[np.argmin(differences)]
    bottom_right = points[np.argmax(sums)]
    bottom_left = points[np.argmax(differences)]

    return np.array(
        [
            top_left,
            top_right,
            bottom_right,
            bottom_left,
        ],
        dtype=np.float32,
    )


def find_card_rectangle(
    foreground: np.ndarray,
) -> np.ndarray | None:
    """
    Find the largest card-like foreground component.

    The candidate must satisfy:
    - reasonable area
    - reasonable ID-card aspect ratio
    - reasonable rectangle fill
    """

    height, width = foreground.shape[:2]

    contours, _ = cv2.findContours(
        foreground,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    image_area = height * width

    candidates = []

    for contour in contours:

        area = cv2.contourArea(contour)

        # Ignore small objects.
        if area < image_area * 0.25:
            continue

        rectangle = cv2.minAreaRect(contour)

        rect_width, rect_height = rectangle[1]

        if rect_width <= 0 or rect_height <= 0:
            continue

        ratio = max(
            rect_width,
            rect_height,
        ) / min(
            rect_width,
            rect_height,
        )

        if not (
            MIN_CARD_RATIO
            <= ratio
            <= MAX_CARD_RATIO
        ):
            continue

        rectangle_area = rect_width * rect_height

        if rectangle_area <= 0:
            continue

        fill_ratio = area / rectangle_area

        if fill_ratio < 0.55:
            continue

        # Prefer large, well-filled rectangles.
        score = area * fill_ratio

        candidates.append(
            (
                score,
                rectangle,
            )
        )

    if not candidates:
        return None

    candidates.sort(
        key=lambda item: item[0],
        reverse=True,
    )

    best_rectangle = candidates[0][1]

    box = cv2.boxPoints(
        best_rectangle
    )

    return box.astype(np.float32)


def detect_card(
    image: np.ndarray,
) -> np.ndarray:
    """
    Main card detection function.

    Returns 4 points in original-image coordinates:

        [
            top_left,
            top_right,
            bottom_right,
            bottom_left,
        ]
    """

    detection_image, scale = resize_for_detection(
        image
    )

    # Detect large dark regions dynamically.
    dark_mask = detect_large_dark_regions(
        detection_image
    )

    # Remove their influence from segmentation.
    clean_image = remove_dark_regions(
        detection_image,
        dark_mask,
    )

    # Create relative GrabCut initialization.
    grabcut_mask = create_grabcut_mask(
        clean_image,
        dark_mask,
    )

    # Segment foreground.
    foreground = grabcut_foreground(
        clean_image,
        grabcut_mask,
    )

    # Clean segmentation.
    foreground = clean_foreground_mask(
        foreground
    )

    # Find card.
    points = find_card_rectangle(
        foreground
    )

    if points is None:
        raise RuntimeError(
            "Could not detect a valid card rectangle."
        )

    # Convert coordinates back to original image size.
    points = points / scale

    points = order_points(
        points
    )

    return points


def draw_debug(
    image: np.ndarray,
    points: np.ndarray,
) -> np.ndarray:
    """
    Draw detected card rectangle and corners.
    """

    output = image.copy()

    polygon = np.round(
        points
    ).astype(np.int32)

    polygon = polygon.reshape(
        (-1, 1, 2)
    )

    # Rectangle
    cv2.polylines(
        output,
        [polygon],
        True,
        (0, 255, 0),
        6,
    )

    labels = [
        "TL",
        "TR",
        "BR",
        "BL",
    ]

    for point, label in zip(
        points,
        labels,
    ):

        x, y = np.round(
            point
        ).astype(int)

        cv2.circle(
            output,
            (x, y),
            14,
            (0, 0, 255),
            -1,
        )

        cv2.putText(
            output,
            label,
            (x + 15, y - 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 0, 0),
            3,
        )

    return output


def main():
    input_path = Path(
        "national_id.jpg"
    )

    output_path = Path(
        "card_detected.jpg"
    )

    image = cv2.imread(
        str(input_path)
    )

    if image is None:
        raise FileNotFoundError(
            f"Could not read image: {input_path}"
        )

    print(
        f"Image size: "
        f"{image.shape[1]} x {image.shape[0]}"
    )

    points = detect_card(
        image
    )

    print("\nDetected card corners:")

    labels = [
        "top_left",
        "top_right",
        "bottom_right",
        "bottom_left",
    ]

    for label, point in zip(
        labels,
        points,
    ):
        x, y = point

        print(
            f"{label}: "
            f"({x:.1f}, {y:.1f})"
        )

    debug_image = draw_debug(
        image,
        points,
    )

    cv2.imwrite(
        str(output_path),
        debug_image,
    )

    print(
        f"\nDebug image saved to: "
        f"{output_path}"
    )


if __name__ == "__main__":
    main()