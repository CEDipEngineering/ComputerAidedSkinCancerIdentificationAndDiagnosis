from fastapi import FastAPI, HTTPException, BackgroundTasks
import uuid
import base64
import cv2
from pydantic import BaseModel
from fastapi.responses import Response
import numpy as np
from starlette.responses import StreamingResponse
import io
from PIL import Image
import os

from cascid.configs import hed_cnf
from cascid.image import image_preprocessing
from cascid.configs.config import DATA_DIR
from cascid.image import HED_segmentation, image_preprocessing
from cascid.configs import config

from app.model import PredictiveModel
model = PredictiveModel(path = DATA_DIR / 'final_models' / 'stacked_01')


app = FastAPI()

db = []

API_DATA = config.DATA_DIR / "api_data"
API_DATA.mkdir(parents=True, exist_ok=True)


API_PREPRO = config.DATA_DIR / "api_prepro"
API_PREPRO.mkdir(parents=True, exist_ok=True)


class UploadItem(BaseModel):
    image_to_base64: str


def preprocess_img(img, fn):
    #path= "/home/fernandofincatti/Pictures/test-hair-removal.jpeg"
    #img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    processed = image_preprocessing.remove_black_hairs_hessian(img)
    _, encoded_img = cv2.imencode('.JPG', processed)
    
    with open(API_PREPRO / fn, "wb") as f:
        f.write(encoded_img)
    return cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)



@app.get("/")
async def test_get():
    return


@app.post("/upload")
async def upload(uploadItem: UploadItem, background_tasks: BackgroundTasks):
    #image_as_bytes = str.encode(uploadItem.image_to_base64)  # convert string to bytes
    img_recovered = base64.b64decode(bytes(uploadItem.image_to_base64, "utf-8"))  # decode base64string
    try:
        fn = str(uuid.uuid4()) + ".jpg"
        with open(API_DATA / fn, "wb") as f:
            f.write(img_recovered)
        db.append(fn)
        #background_tasks.add_task(preprocess_img, fn)

    except Exception as e:
        print(e)
        raise HTTPException(status_code=404, detail="There was an error uploading the file")
    
    return {"path": fn} 

@app.get("/hed_images/{fn}")
async def read_file(fn):
    try:
        img = cv2.cvtColor(cv2.imread(str(API_DATA/fn)), cv2.COLOR_BGR2RGB)
        HED_img = HED_segmentation.HED_segmentation_borders(img)
        img_to_array = Image.fromarray(HED_img.astype("uint8"))
        rawBytes = io.BytesIO()
        img_to_array.save(rawBytes, "JPEG")
        rawBytes.seek(0)
        img_base64 = str(base64.b64encode(rawBytes.read()))
        return {"img_base64": img_base64}
    except Exception as error:
        print(error)
        raise HTTPException(status_code=404, detail="File not found, check your request path")

@app.get("/hed_images_zoom/{fn}")
async def read_file(fn):
    try:
        
        img = cv2.cvtColor(cv2.imread(str(API_DATA/fn)), cv2.COLOR_BGR2RGB)
        h,w,_ = img.shape
        circle_width = int( h*0.2)
        img = img[int(h/2) - circle_width : int(h/2) + circle_width, int(w/2) - circle_width : int(w/2) + circle_width]
        img = cv2.resize(img, (300,300))
        cv2.imwrite(str(API_PREPRO /"original.jpeg"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        img_hariless = preprocess_img(img, fn)
        HED_img = HED_segmentation.HED_segmentation_borders(img_hariless)
        img_to_array = Image.fromarray(HED_img.astype("uint8"))
        rawBytes = io.BytesIO()
        img_to_array.save(rawBytes, "JPEG")
        rawBytes.seek(0)
        img_base64 = str(base64.b64encode(rawBytes.read()))
        return {"img_base64": img_base64}
    except Exception as error:
        print(error)
        raise HTTPException(status_code=404, detail="File not found, check your request path")

@app.get("/images/{fn}")
async def read_file(fn, 
    smoke: bool,
    drink: bool,
    pesticide: bool,
    cancer_history: bool,
    skin_cancer_history: bool,
    age: int
    ):

    # ['smoke', 'drink', 'skin_cancer_history', 'cancer_history', 'age','pesticide'] Use in this order

    metadata = [[
        int(smoke),
        int(drink),
        int(skin_cancer_history),
        int(cancer_history),
        age,
        int(pesticide)
    ]]

    try:
        image = cv2.cvtColor(cv2.imread(str(API_DATA/fn)), cv2.COLOR_BGR2RGB)
        #,img_hariless = preprocess_img(image)
        report = model.produce_report(image, metadata)
        return {"report" : report}
    except Exception as e:
        print(e)
        raise HTTPException(status_code=404, detail="File not found, check your request path")