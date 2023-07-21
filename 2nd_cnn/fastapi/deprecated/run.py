import fastapi
import os
import anomalib
from fastapi import FastAPI, File, UploadFile

from anomalib.config import get_configurable_parameters
from anomalib.data.inference import InferenceDataset
from anomalib.data.utils import InputNormalizationMethod, get_transforms
from anomalib.models import get_model
from anomalib.utils.callbacks import get_callbacks
from pytorch_lightning import Trainer
import torch.utils.data as DataLoader

from fastapi import FastAPI, File, UploadFile
from typing import List
import os
from inference import infer_image

import torch
import torchvision.transforms as transforms
from PIL import Image
from argparse import ArgumentParser, Namespace
from pathlib import Path

from pytorch_lightning import Trainer
from torch.utils.data import DataLoader

from anomalib.config import get_configurable_parameters
from anomalib.data.inference import InferenceDataset
from anomalib.data.utils import InputNormalizationMethod, get_transforms
from anomalib.models import get_model
from anomalib.utils.callbacks import get_callbacks



app = fastapi.FastAPI()


@app.post("/inference")
async def inference(file: bytes = File(...)):
    """Perform inference on an image."""

    config = get_configurable_parameters(config_path="/home/i4624/vscode/gitclone/SWbootProject_2023-7/2nd_cnn/fastapi/results/patchcore/mvtec/hazelnut/run/config.yaml")
    config.trainer.resume_from_checkpoint = "/home/i4624/vscode/gitclone/SWbootProject_2023-7/2nd_cnn/fastapi/results/patchcore/mvtec/hazelnut/run/weights/model.ckpt "
    config.visualization.show_images = False
    config.visualization.mode = "full"

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

    # create the dataset
    dataset = InferenceDataset(
        ["/upload/image.jpg"], image_size=tuple(config.dataset.image_size), transform=transform  # type: ignore
    )
    dataloader = DataLoader(dataset)

    # generate predictions
    trainer.predict(model=model, dataloaders=[dataloader])

    # save the inference results
    file_path = os.path.join("./infer", "inference.jpg")
    model.save_inference_results(file_path)

    return {"file_path": file_path}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
