from fastapi import FastAPI, File, UploadFile
from typing import List
import os
from inference import infer_image


app = FastAPI()

import torch
import torchvision.transforms as transforms
from PIL import Image

def load_model(model_path):
    # Load your PyTorch model from the checkpoint file
    model = YourModelClass()  # Replace YourModelClass with the actual class of your model
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

@app.get("/")
def read_root():
  return { "Hello": "World" }

@app.post("/files/")
async def create_files(files: List[bytes] = File(...)):
    return {"file_sizes": [len(file) for file in files]}

@app.post("/uploadfiles")
async def create_upload_files(files: List[UploadFile] = File(...), filename: str = Query(...)):
    UPLOAD_DIRECTORY = "./input"
    for file in files:
        contents = await file.read()
        with open(os.path.join(UPLOAD_DIRECTORY, file.filename), "wb") as fp:
            fp.write(contents)

    # Process the ML model with the provided filename
    model_weights_path = "/home/i4624/vscode/gitclone/SWbootProject_2023-7/2nd_cnn/fastapi/results/patchcore/mvtec/hazelnut/run/weights/model.ckpt"
    result = infer_image(os.path.join(UPLOAD_DIRECTORY, filename), model_weights_path)

    return {"filenames": [file.filename for file in files], "result": result}


