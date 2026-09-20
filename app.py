import os
import shutil
import uuid
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, HTTPException
from preprocess.card_ocr import process_image, get_ocr

app = FastAPI(title="Egyptian National ID OCR API")

# Ensure upload directory exists
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

# Pre-initialize OCR to avoid latency on first request
print("Initializing PaddleOCR...")
ocr_instance = get_ocr()
print("OCR ready.")

@app.get("/")
def root():
    return {"message": "Egyptian National ID OCR API is running"}

@app.post("/extract")
async def extract_data(file: UploadFile = File(...)):
    """
    Upload an ID image and extract Name, National ID, and Address.
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    # Generate a unique filename to avoid collisions
    file_id = str(uuid.uuid4())
    extension = Path(file.filename).suffix
    temp_path = UPLOAD_DIR / f"{file_id}{extension}"

    try:
        # Save uploaded file
        with temp_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Process image through the pipeline
        data = process_image(temp_path, ocr=ocr_instance)
        
        return data

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    finally:
        # Cleanup: remove temp file
        if temp_path.exists():
            os.remove(temp_path)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
