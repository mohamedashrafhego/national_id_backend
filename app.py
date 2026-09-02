from fastapi import FastAPI

app = FastAPI()


@app.get("/")
def root():
    return {
        "message": "National ID OCR API moheyyyy"
    }