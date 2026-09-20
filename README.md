# Egyptian National ID Detector

This project provides a Python-based utility to detect, straighten (perspective correction), and normalize Egyptian National ID cards from images. It uses OpenCV's GrabCut for segmentation and contour analysis for corner detection.

## Features
- **Automatic Detection**: Locates the ID card in a complex background.
- **Perspective Correction**: Straightens the card to a flat, top-down view.
- **Normalization**: Resizes the card to a consistent 1400x840 resolution for downstream OCR tasks.
- **Debug Output**: Generates annotated images showing detected corners and masks.

## Installation

1. Clone the repository or download the script.
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Basic Usage
Place your image as `national_id.jpg` in the project root and run:
```bash
python preprocess/card_detector.py
```

### Advanced Usage (CLI)
You can specify custom input and output paths:
```bash
python preprocess/card_detector.py --input path/to/id_card.jpg --output my_debug_results
```

## How It Works
1. **Resizing**: The image is downscaled for faster processing while maintaining precision.
2. **GrabCut Segmentation**: Initial background/foreground separation using image borders as a prior.
3. **Component Analysis**: The script identifies the largest rectangular component that matches the aspect ratio of an Egyptian ID.
4. **Corner Extraction**: Uses contour approximation to find the four exact corners, even if the card is slightly tilted.
5. **Warping**: Applies a perspective transform to create a straightened version of the card.

## License
MIT
