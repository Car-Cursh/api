from typing import Annotated

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse, StreamingResponse

import numpy as np
import cv2
from io import BytesIO

app = FastAPI()

@app.get("/")
async def main():
    return FileResponse('view/index.html')



@app.post("/files/")
async def create_file(file: Annotated[bytes, File()]):
    return {"file_size": len(file)}


@app.post("/upload")
async def create_upload_file(file: UploadFile):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Convert back to bytes
    _, encoded_img = cv2.imencode('.PNG', gray)
    byte_io = BytesIO(encoded_img.tobytes())

    return StreamingResponse(BytesIO(encoded_img.tobytes()), media_type="image/png")