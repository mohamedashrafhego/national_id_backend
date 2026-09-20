
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
    use_angle_cls=True,
    show_log=False
)

print("PaddleOCR initialized!")

print("Running OCR...")

# paddleocr 2.x returns result as a list of lists of lines
result = ocr.ocr("national_id.jpg", cls=True)


# --------------------------------------------------
# Convert PaddleOCR result to OCRItem objects
# --------------------------------------------------

items = []


for page in result:
    if page is None:
        continue
    for line in page:
        poly = line[0]
        text, confidence = line[1]

        # Convert polygon [[x1,y1], [x2,y2], [x3,y3], [x4,y4]] to [left, top, right, bottom]
        left = min(p[0] for p in poly)
        right = max(p[0] for p in poly)
        top = min(p[1] for p in poly)
        bottom = max(p[1] for p in poly)

        item = OCRItem(
            text=text,
            confidence=confidence,
            box=[left, top, right, bottom],
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
