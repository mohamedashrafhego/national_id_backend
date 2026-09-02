
from paddleocr import PaddleOCR

from ocr_item import OCRItem

from egyptian_id_extractor import (
    extract_national_id,
    extract_full_name_from_crop,
    extract_address,
    extract_serial_number,
)


print("TEST STARTED")

print("Initializing PaddleOCR...")

ocr = PaddleOCR(
    lang="ar",
)

print("PaddleOCR initialized!")

print("Running OCR...")

result = ocr.predict("national_id.jpg")


# --------------------------------------------------
# Convert PaddleOCR result to OCRItem objects
# --------------------------------------------------

items = []


for page in result:

    texts = page.get("rec_texts", [])
    scores = page.get("rec_scores", [])
    boxes = page.get("rec_boxes", [])

    for index in range(len(texts)):

        text = texts[index]
        score = scores[index]
        box = boxes[index]

        item = OCRItem(
            text=text,
            confidence=score,
            box=box,
        )

        items.append(item)


# --------------------------------------------------
# Extract Egyptian ID Front data
# --------------------------------------------------

print()
print("=" * 80)
print("EGYPTIAN ID FRONT")
print("=" * 80)


national_id = extract_national_id(items)
full_name = extract_full_name_from_crop(
    ocr,
    "national_id.jpg"
)
address = extract_address(items)
serial_number = extract_serial_number(items)


# --------------------------------------------------
# Print extracted data
# --------------------------------------------------

print()
print("Full Name:")
print(full_name)

print()
print("National ID:")
print(national_id)


print()
print("Address:")
print(address)


print()
print("Serial Number:")
print(serial_number)


print()
print("=" * 80)
print("TEST FINISHED")
print("=" * 80)