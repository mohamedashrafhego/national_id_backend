import cv2

image = cv2.imread("national_id.jpg")

if image is None:
    print("ERROR: Could not load national_id.jpg")
    exit()

print("Image loaded")
print("Size:", image.shape[1], "x", image.shape[0])

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

blurred = cv2.GaussianBlur(
    gray,
    (5, 5),
    0,
)

edges = cv2.Canny(
    blurred,
    30,
    100,
)

# Connect broken edges
kernel = cv2.getStructuringElement(
    cv2.MORPH_RECT,
    (9, 9),
)

closed = cv2.morphologyEx(
    edges,
    cv2.MORPH_CLOSE,
    kernel,
)

cv2.imwrite("debug_edges.jpg", edges)
cv2.imwrite("debug_closed.jpg", closed)

print("Saved:")
print(" - debug_edges.jpg")
print(" - debug_closed.jpg")

# Find contours
contours, _ = cv2.findContours(
    closed,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE,
)

print()
print("Contours found:", len(contours))

# Show largest contours
contours = sorted(
    contours,
    key=cv2.contourArea,
    reverse=True,
)

for index, contour in enumerate(contours[:10]):

    area = cv2.contourArea(contour)

    perimeter = cv2.arcLength(
        contour,
        True,
    )

    approximation = cv2.approxPolyDP(
        contour,
        0.02 * perimeter,
        True,
    )

    print(
        f"Contour {index}: "
        f"area={area:.0f}, "
        f"points={len(approximation)}"
    )