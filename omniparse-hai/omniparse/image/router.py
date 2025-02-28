from fastapi import UploadFile, File, HTTPException, APIRouter, Form
from fastapi.responses import JSONResponse
from omniparse import get_shared_state
from omniparse.image import parse_image, process_image
from omniparse.models import responseDocument
from PIL import Image
import requests
import io

image_router = APIRouter()
model_state = get_shared_state()

def read_image_url(url):
    img = Image.open(requests.get(url, stream=True).raw)
    img_bytes = io.BytesIO()
    img.save(img_bytes, format='PNG')
    img_bytes = img_bytes.getvalue()
    return img_bytes

@image_router.post("/image")
async def parse_image_endpoint(file: UploadFile = File(...)):
    try:
        file_bytes = await file.read()
        result: responseDocument = parse_image(file_bytes, model_state)
        return JSONResponse(content=result.model_dump())

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@image_router.post("/process_image")
async def process_image_route(image: UploadFile = File(...), task: str = Form(...)):
    try:
        file_bytes = await image.read()
        result: responseDocument = process_image(file_bytes, task, model_state)
        return JSONResponse(content=result.model_dump())

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@image_router.get("/process_url")
async def process_image_url(url: str):
    try:
        file_bytes = read_image_url(url)
        task = "OCR with Region"
        result: responseDocument = process_image(file_bytes, task, model_state)
        return JSONResponse(content=result.model_dump())

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
