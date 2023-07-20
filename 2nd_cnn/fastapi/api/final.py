from argparse import Namespace
import uvicorn

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import FileResponse

import torch
import io
import os 

from typing import List
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader

from anomalib.config import get_configurable_parameters
from anomalib.data.inference import InferenceDataset
from anomalib.data.utils import InputNormalizationMethod, get_transforms
from anomalib.models import get_model
from anomalib.utils.callbacks import get_callbacks

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/output", StaticFiles(directory="output"), name="output")
templates = Jinja2Templates(directory="templates")



@app.get("/predict", response_class=HTMLResponse)
async def upload_image_form():
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Image Upload</title>
    </head>
    <body>
        <h1>Upload an Image</h1>
        <form action="/predict/" method="post" enctype="multipart/form-data">
            <input type="file" name="file">
            <input type="submit" value="Upload">
        </form>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)


@app.get("/output/{image_path}")
async def get_result_image(image_path: str):
    # Since the {image_path} contains "input/", we need to adjust the file path
    # image_relative_path = image_path.replace("input/", "output/input/")
    image_file_path = os.path.join('/home/i4624/vscode/gitclone/SWbootProject_2023-7/2nd_cnn/fastapi/api/output'+{image_path})

    return FileResponse(image_file_path)

def infer(args: Namespace):
    """Run inference."""
    config = get_configurable_parameters(config_path=args.config)
    config.trainer.resume_from_checkpoint = str(args.weights)
    config.visualization.show_images = args.show
    config.visualization.mode = args.visualization_mode
    if args.output:  # overwrite save path
        config.visualization.save_images = True
        config.visualization.image_save_path = args.output
    else:
        config.visualization.save_images = False

    # create model and trainer
    model = get_model(config)
    callbacks = get_callbacks(config)
    trainer = Trainer(callbacks=callbacks, **config.trainer)

    # get the transforms
    transform_config = config.dataset.transform_config.eval if "transform_config" in config.dataset.keys() else None
    image_size = (config.dataset.image_size[0], config.dataset.image_size[1])
    center_crop = config.dataset.get("center_crop")
    if center_crop is not None:
        center_crop = tuple(center_crop)
    normalization = InputNormalizationMethod(config.dataset.normalization)
    transform = get_transforms(
        config=transform_config, image_size=image_size, center_crop=center_crop, normalization=normalization
    )

    # Create the dataset with the custom data loader function
    dataset = InferenceDataset(
        args.input,
        transform=transform,
        image_size=tuple(config.dataset.image_size),
    )
    dataloader = DataLoader(dataset)

    # generate predictions
    predictions = trainer.predict(model=model, dataloaders=[dataloader])
    
    return predictions


@app.post("/predict/", response_class=HTMLResponse)
async def predict_image(file: UploadFile = File(...)):
    # Save the uploaded image content to the 'input' directory
    file_path = os.path.join('./input', file.filename)
    with open(file_path, "wb") as f:
        f.write(await file.read())

    args = Namespace(
        config='config.yaml',
        weights='model.ckpt',
        input='./input',
        output='./output',
        visualization_mode="full",
        show=False,
    )
    predictions = infer(args)

    image_paths = [pred['image_path'][0] for pred in predictions]

   # Construct the HTML content to display the predicted images
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Predicted Images</title>
    </head>
    <body>
        <h1>Predicted Images</h1>
    """

    for image_path in image_paths:
        html_content += f'<img src="/output/{image_path}" alt="predicted image"><br>'

    html_content += """
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

if __name__ == "__main__":
    uvicorn.run("fastapi_server:app", host="0.0.0.0", port=8000)
