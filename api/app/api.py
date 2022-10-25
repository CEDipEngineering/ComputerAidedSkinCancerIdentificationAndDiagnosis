from fastapi import FastAPI, HTTPException, BackgroundTasks
import uuid
import base64
from cascid.configs import config
import cv2
from pydantic import BaseModel
from cascid.image import HED_segmentation, image_preprocessing
from fastapi.responses import Response
import numpy as np
from cascid.configs import hed_cnf
from cascid.image import image_preprocessing
from starlette.responses import StreamingResponse
import io
import os


from app.model import PredictiveModel
model = PredictiveModel()


app = FastAPI()

db = []

API_DATA = config.DATA_DIR / "api_data"
API_DATA.mkdir(parents=True, exist_ok=True)


API_PREPRO = config.DATA_DIR / "api_prepro"
API_PREPRO.mkdir(parents=True, exist_ok=True)


class UploadItem(BaseModel):
    image_to_base64: str


def preprocess_img(fn : str):
    img = cv2.cvtColor(cv2.imread(str(API_DATA/fn)), cv2.COLOR_BGR2RGB)[:,:,::-1]
    processed = image_preprocessing.adaptive_hair_removal2(img)
    _, encoded_img = cv2.imencode('.JPG', processed)
    
    with open(API_PREPRO / fn, "wb") as f:
        f.write(encoded_img)


@app.get("/")
async def test_get():
    return


@app.post("/upload")
async def upload(uploadItem: UploadItem, background_tasks: BackgroundTasks):
    image_as_bytes = str.encode(uploadItem.image_to_base64)  # convert string to bytes
    img_recovered = base64.b64decode(bytes(uploadItem.image_to_base64, "utf-8"))  # decode base64string
    try:
        fn = str(uuid.uuid4()) + ".jpg"
        with open(API_DATA / fn, "wb") as f:
            f.write(img_recovered)
        db.append(fn)
        background_tasks.add_task(preprocess_img, fn)

    except Exception as e:
        print(e)
        raise HTTPException(status_code=404, detail="There was an error uploading the file")
    
    return {"path": fn} 



@app.get("/hed_images/{fn}")
async def read_file(fn):
    try:
        print()
        print(str(API_PREPRO/fn))
        print()
        try:
            img = cv2.cvtColor(cv2.imread(str(API_PREPRO/fn)), cv2.COLOR_BGR2RGB)
        except: # preprocessing not ready yet
            img = cv2.cvtColor(cv2.imread(str(API_DATA/fn)), cv2.COLOR_BGR2RGB)
        # prepro = image_preprocessing.red_band_unsharp(img)

        print()
        print(HED_segmentation.__file__)
        print()

        HED_img = HED_segmentation.HED_segmentation_borders(img)
        _, encoded_img = cv2.imencode('.jpg', HED_img)
        return StreamingResponse(io.BytesIO(encoded_img.tobytes()), media_type="image/png")

    except Exception:
        raise HTTPException(status_code=404, detail="File not found, check your request path")



@app.get("/images/{fn}")
async def read_file(fn):
    try:
        image = cv2.cvtColor(cv2.imread(str(API_DATA/fn)), cv2.COLOR_BGR2RGB)
        report = model.produce_report(image)
        return {"report" : report}
    except Exception:
        raise HTTPException(status_code=404, detail="File not found, check your request path")
    return {"prediction": str(pred)}
