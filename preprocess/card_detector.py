from pathlib import Path

import cv2
import numpy as np


INPUT_PATH = Path("national_id.jpg")
OUTPUT_DIR = Path("debug_card_detection")


# ============================================================
# Configuration
# ============================================================

MAX_SIDE = 1600

# How much of the image border is considered
# definite background.
BORDER_RATIO = 0.04

# Egyptian National ID approximate width / height ratio.
# We use a broad range because perspective can change it.
MIN_ASPECT_RATIO = 1.20
MAX_ASPECT_RATIO = 2.20

# Candidate area relative to complete image.
MIN_AREA_RATIO = 0.20
MAX_AREA_RATIO = 0.95

# GrabCut iterations.
GRABCUT_ITERATIONS = 8


# ============================================================
# Image utilities
# ============================================================

def resize_for_detection(image: np.ndarray):
    """
    Resize image only for detection speed.

    Returns:
        resized_image
        scale
    """

    height, width = image.shape[:2]

    largest_side = max(
        height,
        width,
    )

    if largest_side <= MAX_SIDE:
        return image.copy(), 1.0

    scale = MAX_SIDE / largest_side

    resized = cv2.resize(
        image,
        None,
        fx=scale,
        fy=scale,
        interpolation=cv2.INTER_AREA,
    )

    return resized, scale


# ============================================================
# GrabCut
# ============================================================

def create_grabcut_mask(
    image: np.ndarray,
):
    """
    Create an initial GrabCut mask.

    The image border is definite background.
    The rest is probable foreground.

    This is dynamic and does not depend on card coordinates.
    """

    height, width = image.shape[:2]

    mask = np.full(
        (height, width),
        cv2.GC_PR_FGD,
        dtype=np.uint8,
    )

    border = max(
        5,
        int(
            min(height, width)
            * BORDER_RATIO
        ),
    )

    # Top
    mask[:border, :] = cv2.GC_BGD

    # Bottom
    mask[-border:, :] = cv2.GC_BGD

    # Left
    mask[:, :border] = cv2.GC_BGD

    # Right
    mask[:, -border:] = cv2.GC_BGD

    return mask


def run_grabcut(
    image: np.ndarray,
):
    """
    Run GrabCut using only image-border information.
    """

    mask = create_grabcut_mask(
        image
    )

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
        GRABCUT_ITERATIONS,
        cv2.GC_INIT_WITH_MASK,
    )

    # Definite foreground OR probable foreground.
    foreground = np.where(
        (
            (mask == cv2.GC_FGD)
            |
            (mask == cv2.GC_PR_FGD)
        ),
        255,
        0,
    ).astype(np.uint8)

    return foreground


# ============================================================
# Mask cleanup
# ============================================================

def clean_foreground_mask(
    mask: np.ndarray,
):
    """
    Remove small noise and connect fragmented regions.
    """

    height, width = mask.shape

    kernel_size = max(
        5,
        int(
            min(height, width)
            * 0.015
        ),
    )

    # Make kernel odd.
    if kernel_size % 2 == 0:
        kernel_size += 1

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (
            kernel_size,
            kernel_size,
        ),
    )

    # Close gaps.
    closed = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=2,
    )

    # Remove small holes.
    filled = cv2.morphologyEx(
        closed,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=2,
    )

    return filled


# ============================================================
# Debug mask
# ============================================================

def save_mask_debug(
    mask: np.ndarray,
    path: Path,
):
    cv2.imwrite(
        str(path),
        mask,
    )


# ============================================================
# Connected Components
# ============================================================

def get_components(
    mask: np.ndarray,
):
    """
    Return connected components with geometry information,
    along with the raw label image so a specific component's
    mask can be rebuilt later (needed for contour extraction).
    """

    height, width = mask.shape

    total_area = float(
        height * width
    )

    number_labels, labels, stats, centroids = (
        cv2.connectedComponentsWithStats(
            mask,
            connectivity=8,
        )
    )

    components = []

    for index in range(
        1,
        number_labels,
    ):
        x = int(stats[index, cv2.CC_STAT_LEFT])
        y = int(stats[index, cv2.CC_STAT_TOP])

        component_width = int(
            stats[index, cv2.CC_STAT_WIDTH]
        )

        component_height = int(
            stats[index, cv2.CC_STAT_HEIGHT]
        )

        area = int(
            stats[index, cv2.CC_STAT_AREA]
        )

        if component_height <= 0:
            continue

        if component_width <= 0:
            continue

        area_ratio = (
            area / total_area
        )

        bbox_ratio = (
            component_width
            / component_height
        )

        bbox_area = (
            component_width
            * component_height
        )

        rectangularity = (
            area / bbox_area
            if bbox_area > 0
            else 0
        )

        components.append(
            {
                "label": index,
                "x": x,
                "y": y,
                "width": component_width,
                "height": component_height,
                "area": area,
                "area_ratio": area_ratio,
                "ratio": bbox_ratio,
                "rectangularity": rectangularity,
            }
        )

    return components, labels


# ============================================================
# Candidate scoring
# ============================================================

def score_candidate(
    candidate,
    image_shape,
):
    """
    Score how likely a connected component is to be
    the ID card.
    """

    height, width = image_shape[:2]

    area_ratio = candidate["area_ratio"]
    ratio = candidate["ratio"]
    rectangularity = candidate["rectangularity"]

    score = 0.0

    # --------------------------------------------------------
    # Area
    # --------------------------------------------------------

    if (
        MIN_AREA_RATIO
        <= area_ratio
        <= MAX_AREA_RATIO
    ):
        score += 35

    else:
        # Distance from allowed area range.
        if area_ratio < MIN_AREA_RATIO:
            distance = (
                MIN_AREA_RATIO
                - area_ratio
            )
        else:
            distance = (
                area_ratio
                - MAX_AREA_RATIO
            )

        score -= min(
            30,
            distance * 100,
        )

    # --------------------------------------------------------
    # Aspect ratio
    # --------------------------------------------------------

    if (
        MIN_ASPECT_RATIO
        <= ratio
        <= MAX_ASPECT_RATIO
    ):
        score += 35

        # Prefer something around
        # a typical ID card ratio.
        target_ratio = 1.59

        ratio_error = abs(
            ratio - target_ratio
        )

        score += max(
            0,
            15 - ratio_error * 15,
        )

    else:
        score -= 30

    # --------------------------------------------------------
    # Rectangularity
    # --------------------------------------------------------

    # A card should occupy a large portion
    # of its bounding rectangle.
    score += min(
        20,
        rectangularity * 20,
    )

    # --------------------------------------------------------
    # Size
    # --------------------------------------------------------

    candidate_width = candidate["width"]
    candidate_height = candidate["height"]

    width_ratio = (
        candidate_width / width
    )

    height_ratio = (
        candidate_height / height
    )

    if width_ratio > 0.50:
        score += 5

    if height_ratio > 0.40:
        score += 5

    # --------------------------------------------------------
    # Border touching penalty
    # --------------------------------------------------------

    x = candidate["x"]
    y = candidate["y"]

    touches_left = x <= width * 0.01
    touches_top = y <= height * 0.01

    touches_right = (
        x + candidate_width
        >= width * 0.99
    )

    touches_bottom = (
        y + candidate_height
        >= height * 0.99
    )

    border_count = sum(
        [
            touches_left,
            touches_top,
            touches_right,
            touches_bottom,
        ]
    )

    # The card should not be the complete image.
    score -= border_count * 15

    return score


def find_best_candidate(
    components,
    image_shape,
):
    """
    Select the best card candidate.
    """

    scored = []

    for candidate in components:

        score = score_candidate(
            candidate,
            image_shape,
        )

        candidate_copy = (
            candidate.copy()
        )

        candidate_copy["score"] = score

        scored.append(
            candidate_copy
        )

    scored.sort(
        key=lambda item: item["score"],
        reverse=True,
    )

    return scored


# ============================================================
# Corners
# ============================================================

def get_component_mask(
    labels: np.ndarray,
    label: int,
):
    """
    Rebuild a binary mask for a single connected component.
    """

    return np.where(
        labels == label,
        255,
        0,
    ).astype(np.uint8)


# Contour polygon approximation is tried at increasing epsilon
# ratios (relative to the contour perimeter) until a convex
# quadrilateral is found.
CORNER_EPSILON_RATIOS = (
    0.01,
    0.015,
    0.02,
    0.025,
    0.03,
    0.04,
    0.05,
)


def find_contour_corners(
    mask: np.ndarray,
):
    """
    Find the card's real 4 corners from its component mask.

    Component Mask -> Find Contour -> approxPolyDP -> 4 corners

    Returns an (4, 2) array of unordered corners, or None if no
    convex quadrilateral could be extracted from the contour.
    """

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    if not contours:
        return None

    contour = max(
        contours,
        key=cv2.contourArea,
    )

    perimeter = cv2.arcLength(
        contour,
        True,
    )

    if perimeter <= 0:
        return None

    for epsilon_ratio in CORNER_EPSILON_RATIOS:

        approx = cv2.approxPolyDP(
            contour,
            epsilon_ratio * perimeter,
            True,
        )

        if len(approx) != 4:
            continue

        if not cv2.isContourConvex(approx):
            continue

        return approx.reshape(4, 2).astype(np.float32)

    # Noisy outline: fall back to the minimum-area rectangle of
    # the actual contour (still real geometry, not a fixed angle).
    rect = cv2.minAreaRect(contour)

    return cv2.boxPoints(rect)


def get_rotated_corners(
    candidate,
):
    """
    Get 4 corners using minAreaRect.

    This is more flexible than simply returning
    the bounding-box corners.
    """

    x = candidate["x"]
    y = candidate["y"]

    width = candidate["width"]
    height = candidate["height"]

    rect = (
        (
            x + width / 2,
            y + height / 2,
        ),
        (
            width,
            height,
        ),
        0,
    )

    corners = cv2.boxPoints(
        rect
    )

    return order_corners(
        corners
    )


def order_corners(
    points,
):
    """
    Return corners in this order:

        top-left
        top-right
        bottom-right
        bottom-left
    """

    points = np.asarray(
        points,
        dtype=np.float32,
    )

    ordered = np.zeros(
        (4, 2),
        dtype=np.float32,
    )

    sums = points.sum(
        axis=1
    )

    differences = (
        points[:, 1]
        - points[:, 0]
    )

    ordered[0] = points[
        np.argmin(sums)
    ]

    ordered[2] = points[
        np.argmax(sums)
    ]

    ordered[1] = points[
        np.argmin(differences)
    ]

    ordered[3] = points[
        np.argmax(differences)
    ]

    return ordered


# ============================================================
# Debug drawing
# ============================================================

def draw_card_debug(
    image,
    corners,
):
    output = image.copy()

    points = corners.astype(
        np.int32
    )

    # Card polygon
    cv2.polylines(
        output,
        [points],
        True,
        (0, 255, 0),
        5,
    )

    labels = [
        "top_left",
        "top_right",
        "bottom_right",
        "bottom_left",
    ]

    for index, (
        point,
        label,
    ) in enumerate(
        zip(points, labels)
    ):

        x, y = point

        cv2.circle(
            output,
            (x, y),
            12,
            (0, 0, 255),
            -1,
        )

        cv2.putText(
            output,
            f"{index}: {label}",
            (
                x + 15,
                y - 15,
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 0, 0),
            2,
        )

    return output


def draw_candidates(
    image,
    candidates,
):
    output = image.copy()

    for index, candidate in enumerate(
        candidates[:10]
    ):

        x = candidate["x"]
        y = candidate["y"]

        width = candidate["width"]
        height = candidate["height"]

        score = candidate["score"]

        cv2.rectangle(
            output,
            (
                x,
                y,
            ),
            (
                x + width,
                y + height,
            ),
            (0, 255, 255),
            2,
        )

        cv2.putText(
            output,
            f"{index}: score={score:.1f}",
            (
                x,
                max(
                    20,
                    y - 10,
                ),
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2,
        )

    return output


# ============================================================
# Main detector
# ============================================================

def detect_card(
    image: np.ndarray,
):
    """
    Main dynamic card detection API.

    Returns:

        np.ndarray shape (4, 2)

    ordered as:

        top-left
        top-right
        bottom-right
        bottom-left
    """

    original_height, original_width = (
        image.shape[:2]
    )

    working_image, scale = (
        resize_for_detection(image)
    )

    # --------------------------------------------------------
    # 1. GrabCut
    # --------------------------------------------------------

    foreground_mask = run_grabcut(
        working_image
    )

    # --------------------------------------------------------
    # 2. Clean mask
    # --------------------------------------------------------

    foreground_mask = (
        clean_foreground_mask(
            foreground_mask
        )
    )

    # --------------------------------------------------------
    # 3. Connected components
    # --------------------------------------------------------

    components, labels = get_components(
        foreground_mask
    )

    print(
        f"Components found: "
        f"{len(components)}"
    )

    # --------------------------------------------------------
    # 4. Score candidates
    # --------------------------------------------------------

    candidates = find_best_candidate(
        components,
        working_image.shape,
    )

    print(
        "\nCard candidates:"
    )

    for index, candidate in enumerate(
        candidates[:10]
    ):

        print(
            f"{index}: "
            f"score={candidate['score']:.2f}, "
            f"area_ratio={candidate['area_ratio']:.3f}, "
            f"bbox=("
            f"{candidate['x']},"
            f"{candidate['y']},"
            f"{candidate['width']},"
            f"{candidate['height']}"
            f"), "
            f"ratio={candidate['ratio']:.3f}, "
            f"rectangularity="
            f"{candidate['rectangularity']:.3f}"
        )

    if not candidates:
        raise RuntimeError(
            "No card candidates found."
        )

    best = candidates[0]

    if best["score"] < 30:
        raise RuntimeError(
            "Could not confidently detect "
            "the ID card."
        )

    # --------------------------------------------------------
    # 5. Get corners from the best component's real outline
    # --------------------------------------------------------

    component_mask = get_component_mask(
        labels,
        best["label"],
    )

    save_mask_debug(
        component_mask,
        OUTPUT_DIR / "component_mask.jpg",
    )

    raw_corners = find_contour_corners(
        component_mask
    )

    if raw_corners is None:
        # No usable contour: fall back to the axis-aligned
        # bounding box so detection never hard-fails.
        corners = get_rotated_corners(
            best
        )
    else:
        corners = order_corners(
            raw_corners
        )

    # --------------------------------------------------------
    # 6. Convert back to original image
    # --------------------------------------------------------

    if scale != 1.0:
        corners /= scale

    # Keep points inside image.
    corners[:, 0] = np.clip(
        corners[:, 0],
        0,
        original_width - 1,
    )

    corners[:, 1] = np.clip(
        corners[:, 1],
        0,
        original_height - 1,
    )

    return corners


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

    print(
        f"Original image size: "
        f"{image.shape[1]} x "
        f"{image.shape[0]}"
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Detection
    # --------------------------------------------------------

    corners = detect_card(
        image
    )

    print(
        "\nDetected card corners:"
    )

    names = [
        "top_left",
        "top_right",
        "bottom_right",
        "bottom_left",
    ]

    for name, point in zip(
        names,
        corners,
    ):

        print(
            f"{name}: "
            f"({point[0]:.1f}, "
            f"{point[1]:.1f})"
        )

    # --------------------------------------------------------
    # Debug output
    # --------------------------------------------------------

    debug = draw_card_debug(
        image,
        corners,
    )

    debug_path = (
        OUTPUT_DIR
        / "card_detected.jpg"
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