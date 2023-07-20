from argparse import Namespace
import uvicorn
from fastapi import FastAPI, UploadFile, File
import torch
import io

from typing import List
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader

from anomalib.config import get_configurable_parameters
from anomalib.data.inference import InferenceDataset
from anomalib.data.utils import InputNormalizationMethod, get_transforms
from anomalib.models import get_model
from anomalib.utils.callbacks import get_callbacks

app = FastAPI()


def infer(args: Namespace, file):
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
        transform=transform,
        image_size=tuple(config.dataset.image_size),
    )
    dataloader = DataLoader(dataset)

    # generate predictions
    predictions = trainer.predict(model=model, dataloaders=[dataloader])
    return predictions

@app.post("/predict/")
async def predict_image(file: UploadFile = File(...)):
    args = Namespace(
        config='config.yaml',
        weights='model_weights.ckpt',
        input=file,
        output=None,
        visualization_mode="simple",
        show=False,
    )
    predictions = infer(args)
    return {"predictions": predictions}

if __name__ == "__main__":
    uvicorn.run("fastapi_server:app", host="0.0.0.0", port=8000)
