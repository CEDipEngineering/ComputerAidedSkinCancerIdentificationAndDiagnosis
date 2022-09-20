from fastapi import FastAPI, HTTPException
from app.model import PredictiveModel
import uuid
import base64
from cascid.configs import config
import cv2
from pydantic import BaseModel

app = FastAPI()

db = []

API_DATA = config.DATA_DIR / "api_data"
API_DATA.mkdir(parents=True, exist_ok=True)

model = PredictiveModel()

class UploadItem(BaseModel):
    image_to_base64: str

@app.get("/")
async def test_get():
    return

@app.post("/upload")
def upload(uploadItem: UploadItem):
    image_as_bytes = str.encode(uploadItem.image_to_base64)  # convert string to bytes
    img_recovered = base64.b64decode(image_as_bytes)  # decode base64string
    try:
        fn = str(uuid.uuid4()) + ".jpg"
        with open(API_DATA / fn, "wb") as f:
            f.write(img_recovered)
        db.append(fn)
    except Exception as e:
        print(e)
        raise HTTPException(status_code=404, detail="There was an error uploading the file")
    
    return {"path": fn} 

@app.get("/images/{fn}")
async def read_file(fn):
    try:
        image = cv2.cvtColor(cv2.imread(str(API_DATA/fn)), cv2.COLOR_BGR2RGB)
        pred = model.predict_proba(image)
        print(pred)
    except Exception:
        raise HTTPException(status_code=404, detail="File not found, check your request path")
    return {"prediction": str(pred)}