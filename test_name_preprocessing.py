
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
from paddleocr import PaddleOCR


print("Initializing PaddleOCR...")

ocr = PaddleOCR(
    lang="ar",
)

print("PaddleOCR initialized!")


# --------------------------------------------------
# Load original image
# --------------------------------------------------

image = Image.open("national_id.jpg")

print("Original image size:", image.size)


# --------------------------------------------------
# Crop name area
# --------------------------------------------------

name_crop = image.crop(
    (
        1500,   # left
        1000,   # top
        3350,   # right
        1500    # bottom
    )
)


# --------------------------------------------------
# Create different preprocessing versions
# --------------------------------------------------

versions = {}


# 1. Original
versions["original"] = name_crop


# 2. Enlarged
width, height = name_crop.size

versions["enlarged"] = name_crop.resize(
    (
        width * 2,
        height * 2
    ),
    Image.Resampling.LANCZOS
)


# 3. Grayscale
gray = ImageOps.grayscale(
    name_crop
)

versions["grayscale"] = gray


# 4. Grayscale + contrast
contrast = ImageEnhance.Contrast(
    gray
).enhance(2.0)

versions["contrast"] = contrast


# 5. Grayscale + sharpen
sharpen = gray.filter(
    ImageFilter.SHARPEN
)

versions["sharpen"] = sharpen


# 6. Grayscale + contrast + sharpen
enhanced = ImageEnhance.Contrast(
    gray
).enhance(2.0)

enhanced = enhanced.filter(
    ImageFilter.SHARPEN
)

versions["enhanced"] = enhanced


# --------------------------------------------------
# Run OCR for every version
# --------------------------------------------------

for name, processed_image in versions.items():

    filename = f"name_{name}.jpg"

    processed_image.save(filename)

    print()
    print("=" * 80)
    print(f"TESTING: {name}")
    print("=" * 80)

    result = ocr.predict(filename)

    for page in result:

        texts = page.get(
            "rec_texts",
            []
        )

        scores = page.get(
            "rec_scores",
            []
        )

        boxes = page.get(
            "rec_boxes",
            []
        )

        for index in range(len(texts)):

            text = texts[index]
            score = scores[index]
            box = boxes[index]

            left, top, right, bottom = box

            center_x = (
                left + right
            ) // 2

            center_y = (
                top + bottom
            ) // 2

            print(
                f"Text       : {text}"
            )

            print(
                f"Confidence : {score:.2f}"
            )

            print(
                f"Center     : "
                f"({center_x}, {center_y})"
            )

            print("-" * 80)


print()
print("=" * 80)
print("ALL TESTS FINISHED")
print("=" * 80)

