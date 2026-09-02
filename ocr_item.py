class OCRItem:

    def __init__(
        self,
        text,
        confidence,
        box,
    ):
        self.text = text
        self.confidence = confidence
        self.box = box

        self.left = box[0]
        self.top = box[1]
        self.right = box[2]
        self.bottom = box[3]

        self.center_x = (self.left + self.right) // 2
        self.center_y = (self.top + self.bottom) // 2

    def __repr__(self):
        return (
            f"OCRItem("
            f"text='{self.text}', "
            f"confidence={self.confidence:.2f}, "
            f"x={self.center_x}, "
            f"y={self.center_y}"
            f")"
        )