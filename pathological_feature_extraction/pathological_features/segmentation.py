import warnings
from typing import Literal, Union, List, Tuple
from functools import lru_cache

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as Ftv
from lesion_counting_myp.hf_hub import download_model
from lesion_counting_myp.constants import Dataset
from lesion_counting_myp.functional import (
    autofit_fundus_resolution,
    reverse_autofit_tensor,
)
from lesion_counting_myp.config import get_normalization


Architecture = Literal["unet"]
EncoderModel = Literal["resnet34"]


def segment(
    image: np.ndarray,
    arch: Architecture = "unet",
    encoder: EncoderModel = "seresnext50_32x4d",
    train_datasets: Union[Dataset, Tuple[Dataset]] = Dataset.ALL,
    image_resolution=1024,
    autofit_resolution=True,
    reverse_autofit=True,
    mean=None,
    std=None,
    return_features=False,
    return_decoder_features=False,
    features_layer=3,
    device: torch.device = "cuda",
    compile: bool = False,
):
    """Segment fundus image into 5 classes: background, CTW, EX, HE, MA

    Args:
        image (np.ndarray):   Fundus image of size HxWx3
        arch (Architecture, optional): Defaults to 'unet'.
        encoder (EncoderModel, optional): Defaults to 'resnest50d'.
        weights (TrainedOn, optional):  Defaults to 'All'.
        image_resolution (int, optional): Defaults to 1024.
        mean (list, optional): Defaults to constants.DEFAULT_NORMALIZATION_MEAN.
        std (list, optional): Defaults to constants.DEFAULT_NORMALIZATION_STD.
        autofit_resolution (bool, optional):  Defaults to True.
        return_features (bool, optional): Defaults to False. If True, returns also the features map of the i-th encoder layer. See features_layer.
        features_layer (int, optional): Defaults to 3. If return_features is True, returns the features map of the i-th encoder layer.
        device (torch.device, optional): Defaults to "cuda".

    Returns:
        torch.Tensor: 5 channel tensor with probabilities of each class (size 5xHxW)
    """
    model = get_model(arch, encoder, train_datasets, device, compile=compile)
    model.eval()
    h, w, c = image.shape
    if autofit_resolution:
        image, roi, transforms = autofit_fundus_resolution(
            image, image_resolution, return_roi=True
        )

    image = (image / 255.0).astype(np.float32)
    tensor = torch.from_numpy(image).permute((2, 0, 1)).unsqueeze(0).to(device)

    if mean is None:
        mean = get_normalization()[0]
    if std is None:
        std = get_normalization()[1]
    tensor = Ftv.normalize(tensor, mean=mean, std=std)

    with torch.inference_mode():
        features = model.encoder(tensor)
        pre_segmentation_features = model.decoder(*features)
        pred = model.segmentation_head(pre_segmentation_features)
        pred = F.softmax(pred, 1)
        if return_features or return_decoder_features:
            assert not reverse_autofit, "reverse_autofit is not compatible with return_features or return_decoder_features"
            out = [pred]
            if return_features:
                out.append(features[features_layer])
            if return_decoder_features:
                out.append(pre_segmentation_features)
            return tuple(out)

    pred = pred.squeeze(0)
    if reverse_autofit and autofit_resolution:
        pred = reverse_autofit_tensor(pred, **transforms)
        all_zeros = ~torch.any(pred, dim=0)  # Find all zeros probabilities
        pred[0, all_zeros] = 1  # Assign them to background
    return pred


@lru_cache(maxsize=2)
def get_model(
    arch: Architecture = "unet",
    encoder: EncoderModel = "resnest50d",
    train_datasets: Union[Dataset, Tuple[Dataset]] = Dataset.ALL,
    device: torch.device = "cuda",
    compile: bool = False,
):
    """Get segmentation model

    Args:
        arch (Architecture, optional): Defaults to 'unet'.
        encoder (EncoderModel, optional):  Defaults to 'resnest50d'.
        weights (TrainedOn, optional):  Defaults to 'All'.
        device (torch.device, optional): Defaults to "cuda".

    Returns:
        nn.Module: Torch segmentation model
    """
    model = download_model(arch, encoder, train_datasets).to(device=device)
    set_dropout(model, initial_value=0.2)
    if compile:
        model.eval()
        with torch.inference_mode():
            model = torch.compile(model)
    return model


def set_dropout(model, initial_value=0.0):
    warnings.warn(f"Setting dropout to {initial_value}")
    for k, v in list(model.named_modules()):
        if "drop" in k.split("."):
            parent_model = model
            for model_name in k.split(".")[:-1]:
                parent_model = getattr(parent_model, model_name)
            setattr(parent_model, "drop", nn.Dropout2d(p=initial_value))

    return model