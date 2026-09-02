
from PIL import Image
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

print("Image size:", image.size)


# --------------------------------------------------
# Crop the name area
# --------------------------------------------------

name_crop = image.crop(
    (
        1500,   # left
        1100,   # top
        3350,   # right
        1450    # bottom
    )
)


name_crop.save("name_crop.jpg")

print("Name crop saved as: name_crop.jpg")


# --------------------------------------------------
# Run OCR on name crop
# --------------------------------------------------

print("Running OCR on name crop...")

result = ocr.predict("name_crop.jpg")


# --------------------------------------------------
# Print result
# --------------------------------------------------

print()
print("=" * 80)
print("NAME CROP OCR RESULT")
print("=" * 80)


for page in result:

    texts = page.get("rec_texts", [])
    scores = page.get("rec_scores", [])

    for index in range(len(texts)):

        print(
            f"#{index + 1}"
        )

        print(
            f"Text       : {texts[index]}"
        )

        print(
            f"Confidence : {scores[index]:.2f}"
        )

        print("-" * 80)


print("=" * 80)
print("TEST FINISHED")
print("=" * 80)

