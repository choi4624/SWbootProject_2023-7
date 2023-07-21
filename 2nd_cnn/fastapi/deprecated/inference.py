# inference.py
from argparse import Namespace
from pathlib import Path

from pytorch_lightning import Trainer
from torch.utils.data import DataLoader

from anomalib.config import get_configurable_parameters
from anomalib.data.inference import InferenceDataset
from anomalib.data.utils import InputNormalizationMethod, get_transforms
from anomalib.models import get_model
from anomalib.utils.callbacks import get_callbacks


def infer_image(image_path: Path, weights_path: Path, output_path: Path = None, visualization_mode: str = "simple"):
    args = Namespace(
        config="/home/i4624/vscode/gitclone/SWbootProject_2023-7/2nd_cnn/anomalib/src/anomalib/models/patchcore/config.yaml",
        weights=weights_path,
        input=image_path,
        output=output_path,
        visualization_mode=visualization_mode,
        show=False,
    )

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

    # create the dataset
    dataset = InferenceDataset(args.input, image_size=tuple(config.dataset.image_size), transform=transform)
    dataloader = DataLoader(dataset)

    # generate predictions
    predictions = trainer.predict(model=model, dataloaders=[dataloader])

    return predictions
