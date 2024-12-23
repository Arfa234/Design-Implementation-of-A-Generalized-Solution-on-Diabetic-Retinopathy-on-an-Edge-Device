from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.utils import draw_segmentation_masks

from lesion_counting_myp.constants import DEFAULT_COLORS


def plot_image(image, figsize=(10, 10), axis="off", title=None):
    """Plot an image"""
    plt.imshow(image)
    plt.gcf().set_size_inches(figsize)
    plt.axis(axis)
    if title:
        plt.title(title)
    plt.show()

def get_segmentation_mask_on_image(
    image: Union[np.ndarray, torch.Tensor],
    mask: torch.Tensor,
    alpha=0.5,
    border_alpha=0.5,
    colors=None,
    border_width=3,
    num_classes=5,
    no_border=False,
):
    if border_width == 0 or border_alpha == 0:
        no_border = True

    if isinstance(image, np.ndarray):
        image = torch.from_numpy(image)
        image = image.to(mask.device)

    if colors is None:
        colors = DEFAULT_COLORS
    if image.shape[0] != 3:
        image = image.permute((2, 0, 1))

    image = (image - image.min()) / (image.max() - image.min())
    image = (255 * image).to(torch.uint8)
    if mask.ndim == 3 and (mask.shape[0] == num_classes):
        mask = mask.unsqueeze(0)

    if mask.ndim == 4:
        mask = torch.argmax(mask, 1)
    mask = F.one_hot(mask, num_classes=num_classes).squeeze(0).permute((2, 0, 1))
    mask[0] = 0  # Remove background
    draw = draw_segmentation_masks(
        image.to(torch.uint8).cpu(),
        mask.to(torch.bool).cpu(),
        alpha=alpha,
        colors=colors,
    )
    if not no_border:
        from kornia.morphology import gradient

        kernel = mask.new_ones((border_width, border_width))
        border = gradient(mask.unsqueeze(0), kernel).squeeze(0)
        border[0] = 0
        draw = draw_segmentation_masks(draw, border.to(torch.bool).cpu(), alpha=1 - border_alpha, colors="white")
    return draw


def plot_image_and_mask(
    image,
    mask,
    alpha=0.5,
    border_alpha=0.8,
    colors=None,
    title=None,
    figsize=(10, 10),
    labels=None,
    save_as=None,
    border_width=3,
    num_classes=5,
    no_border=False,
):
    """Plot image and mask"""

    plt.imshow(
        get_segmentation_mask_on_image(
            image,
            mask,
            alpha,
            border_alpha=border_alpha,
            border_width=border_width,
            colors=colors,
            num_classes=num_classes,
            no_border=no_border,
        )
        .permute((1, 2, 0))
        .cpu()
    )
    plt.axis("off")
    if title:
        plt.title(title)
    plt.gcf().set_size_inches(figsize)
    if labels and colors:
        from matplotlib.patches import Patch

        legend_elements = [Patch(facecolor=c, label=l) for l, c in zip(labels[1:], colors[1:])]
        plt.gca().legend(handles=legend_elements, loc="upper right")

    if save_as:
        plt.savefig(save_as)
    plt.show()
