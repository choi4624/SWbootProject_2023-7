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
from datetime import datetime
import shutil

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

def dir_cleaning():
    # Create the base archive directories if they don't exist
    os.makedirs('./input', exist_ok=True)
    os.makedirs('./output/input', exist_ok=True)

    # Get the current date and time in YYYY-MM-DD format
    current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M")

     # Initialize a counter to keep track of uploads within the same second
    upload_counter = 0

    # Define the base directory names for input and output archives
    base_input_archive_dir = f"./archive/{current_datetime}"
    base_output_archive_dir = f"./archive/{current_datetime}"

    # Create the base archive directories if they don't exist
    os.makedirs(base_input_archive_dir, exist_ok=True)
    os.makedirs(base_output_archive_dir, exist_ok=True)

    # Create the subdirectory names using the upload counter
    input_archive_dir = f"{base_input_archive_dir}/input_{upload_counter}"
    output_archive_dir = f"{base_output_archive_dir}/output_{upload_counter}"

    # Increment the upload counter if the subdirectories already exist
    while os.path.exists(input_archive_dir) or os.path.exists(output_archive_dir):
        upload_counter += 1
        input_archive_dir = f"{base_input_archive_dir}/input_{upload_counter}"
        output_archive_dir = f"{base_output_archive_dir}/output_{upload_counter}"

    # Create the archive directories if they don't exist
    os.makedirs(input_archive_dir, exist_ok=True)
    os.makedirs(output_archive_dir, exist_ok=True)

    # Move the input image to the input archive directory
    shutil.move('./input', input_archive_dir)
    # Move the output image to the output archive directory (if needed)
    shutil.move('./output/input', output_archive_dir)
    

    # Create the base archive directories if they don't exist
    # 옮길 때 디렉터리 단위로 움직이므로 다시 생성해줘야 함 
    os.makedirs('./input', exist_ok=True)
    os.makedirs('./output/input', exist_ok=True)




@app.get("/predict", response_class=HTMLResponse)
async def upload_image_form():

    # cleaning pre trained result of images > move to archive 
    # 아카이브로 던지는 코드, 분 단위로 폴더 분류 
    dir_cleaning()

    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Image Upload</title>
    </head>
    <body>
        <h1>Upload an Image</h1>
        <form action="/predict" method="post" enctype="multipart/form-data">
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


@app.post("/predict", response_class=HTMLResponse)
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
