from pathlib import Path

import cv2
import numpy as np


INPUT_PATH = Path("national_id.jpg")
OUTPUT_DIR = Path("debug_card_detection_v2")

MAX_SIDE = 1400


def resize_image(image: np.ndarray):
    height, width = image.shape[:2]

    scale = min(
        1.0,
        MAX_SIDE / max(height, width),
    )

    if scale == 1.0:
        return image.copy(), 1.0

    resized = cv2.resize(
        image,
        None,
        fx=scale,
        fy=scale,
        interpolation=cv2.INTER_AREA,
    )

    return resized, scale


def create_edge_maps(image: np.ndarray):
    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY,
    )

    # Reduce text-level noise while preserving large edges.
    blurred = cv2.GaussianBlur(
        gray,
        (7, 7),
        0,
    )

    # Sobel gives us horizontal and vertical information.
    sobel_x = cv2.Sobel(
        blurred,
        cv2.CV_32F,
        1,
        0,
        ksize=3,
    )

    sobel_y = cv2.Sobel(
        blurred,
        cv2.CV_32F,
        0,
        1,
        ksize=3,
    )

    abs_x = cv2.convertScaleAbs(
        sobel_x
    )

    abs_y = cv2.convertScaleAbs(
        sobel_y
    )

    magnitude = cv2.magnitude(
        sobel_x,
        sobel_y,
    )

    magnitude = cv2.normalize(
        magnitude,
        None,
        0,
        255,
        cv2.NORM_MINMAX,
    ).astype(np.uint8)

    # Canny with relatively low thresholds.
    canny = cv2.Canny(
        blurred,
        30,
        100,
    )

    return (
        gray,
        abs_x,
        abs_y,
        magnitude,
        canny,
    )


def create_long_edge_map(
    canny: np.ndarray,
):
    height, width = canny.shape

    # Horizontal kernel:
    # connects broken horizontal card edges.
    horizontal_length = max(
        15,
        int(width * 0.12),
    )

    horizontal_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (
            horizontal_length,
            3,
        ),
    )

    horizontal = cv2.morphologyEx(
        canny,
        cv2.MORPH_CLOSE,
        horizontal_kernel,
    )

    # Vertical kernel:
    # connects broken vertical card edges.
    vertical_length = max(
        15,
        int(height * 0.12),
    )

    vertical_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (
            3,
            vertical_length,
        ),
    )

    vertical = cv2.morphologyEx(
        canny,
        cv2.MORPH_CLOSE,
        vertical_kernel,
    )

    combined = cv2.bitwise_or(
        horizontal,
        vertical,
    )

    return (
        horizontal,
        vertical,
        combined,
    )


def detect_lines(
    edge_map: np.ndarray,
):
    lines = cv2.HoughLinesP(
        edge_map,
        rho=1,
        theta=np.pi / 180,
        threshold=80,
        minLineLength=int(
            min(edge_map.shape) * 0.30
        ),
        maxLineGap=int(
            min(edge_map.shape) * 0.05
        ),
    )

    if lines is None:
        return []

    return lines.reshape(
        -1,
        4,
    )


def draw_lines(
    image: np.ndarray,
    lines,
):
    output = image.copy()

    for index, line in enumerate(lines):

        x1, y1, x2, y2 = map(
            int,
            line,
        )

        cv2.line(
            output,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            3,
        )

        cv2.putText(
            output,
            str(index),
            (x1, y1),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2,
        )

    return output


def print_lines(
    lines,
    image_shape,
):
    height, width = image_shape

    print(
        f"\nDetected long lines: {len(lines)}"
    )

    for index, line in enumerate(lines):

        x1, y1, x2, y2 = line

        dx = x2 - x1
        dy = y2 - y1

        length = float(
            np.sqrt(
                dx * dx
                + dy * dy
            )
        )

        angle = np.degrees(
            np.arctan2(
                dy,
                dx,
            )
        )

        if angle < 0:
            angle += 180

        normalized_length = (
            length
            / np.sqrt(
                width * width
                + height * height
            )
        )

        print(
            f"Line {index}: "
            f"({x1},{y1}) → "
            f"({x2},{y2}) "
            f"length={length:.1f} "
            f"angle={angle:.1f}° "
            f"normalized={normalized_length:.3f}"
        )


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
            "Could not read image."
        )

    print(
        f"Original image: "
        f"{image.shape[1]} x "
        f"{image.shape[0]}"
    )

    working_image, scale = resize_image(
        image
    )

    print(
        f"Working image: "
        f"{working_image.shape[1]} x "
        f"{working_image.shape[0]}"
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    (
        gray,
        sobel_x,
        sobel_y,
        magnitude,
        canny,
    ) = create_edge_maps(
        working_image
    )

    cv2.imwrite(
        str(
            OUTPUT_DIR / "01_gray.jpg"
        ),
        gray,
    )

    cv2.imwrite(
        str(
            OUTPUT_DIR / "02_sobel_x.jpg"
        ),
        sobel_x,
    )

    cv2.imwrite(
        str(
            OUTPUT_DIR / "03_sobel_y.jpg"
        ),
        sobel_y,
    )

    cv2.imwrite(
        str(
            OUTPUT_DIR / "04_edge_magnitude.jpg"
        ),
        magnitude,
    )

    cv2.imwrite(
        str(
            OUTPUT_DIR / "05_canny.jpg"
        ),
        canny,
    )

    (
        horizontal,
        vertical,
        combined,
    ) = create_long_edge_map(
        canny
    )

    cv2.imwrite(
        str(
            OUTPUT_DIR / "06_horizontal_edges.jpg"
        ),
        horizontal,
    )

    cv2.imwrite(
        str(
            OUTPUT_DIR / "07_vertical_edges.jpg"
        ),
        vertical,
    )

    cv2.imwrite(
        str(
            OUTPUT_DIR / "08_long_edges.jpg"
        ),
        combined,
    )

    lines = detect_lines(
        combined
    )

    print_lines(
        lines,
        combined.shape,
    )

    line_debug = draw_lines(
        working_image,
        lines,
    )

    cv2.imwrite(
        str(
            OUTPUT_DIR / "09_detected_lines.jpg"
        ),
        line_debug,
    )

    print(
        "\nGenerated files:"
    )

    for path in sorted(
        OUTPUT_DIR.glob("*.jpg")
    ):
        print(path)


if __name__ == "__main__":
    main()