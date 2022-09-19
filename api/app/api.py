from fastapi import FastAPI, Form
from app.model import PredictiveModel
import uuid
import base64
from cascid.configs import config
import cv2

app = FastAPI()

db = []

API_DATA = config.DATA_DIR / "api_data"
API_DATA.mkdir(parents=True, exist_ok=True)

model = PredictiveModel()

@app.post("/upload")
def upload(imagedata: str = Form(...)):
    image_as_bytes = str.encode(imagedata)  # convert string to bytes
    img_recovered = base64.b64decode(image_as_bytes)  # decode base64string
    try:
        fn = str(uuid.uuid4()) + ".jpg"
        with open(API_DATA / fn, "wb") as f:
            f.write(img_recovered)
        db.append(fn)
    except Exception as e:
        print(e)
        return {"message": "There was an error uploading the file"}
    
    return {"path": fn} 


@app.get("/images/{fn}")
async def read_file(fn):
    try:
        image = cv2.cvtColor(cv2.imread(str(API_DATA/fn)), cv2.COLOR_BGR2RGB)
        pred = model.predict_proba(image)
        print(pred)
    except Exception:
        return {"Error" : "File not found, check your request path"}
    return {"prediction": str(pred)}