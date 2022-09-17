from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from starlette.status import HTTP_302_FOUND

app = FastAPI()

@app.get('/', tags=['root'])
async def read_root() -> dict:
    return {"message": "Hello World"}


# async def register_post():
#     # Implementation details ...

#     return RedirectResponse(
#         '/account', # Target path
#         status_code=HTTP_302_FOUND # Code to redirect while changing request from post to get
#     )