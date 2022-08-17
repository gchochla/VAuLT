from typing import Union, Tuple

import torch
from PIL.Image import Image
from torchvision.transforms import transforms


def _get_wh(img: Union[Image, torch.Tensor]) -> Tuple[int, int]:
    """Return (width, height) of image, be it PIL Image or Pytorch Tensor."""
    if isinstance(img, Image):
        return img.size

    _, h, w = img.shape
    return w, h


def vilt_safe_image_preprocess():
    """Returns a `torchvision` transformation that deals with
    extreme ratios of width and height, which cause the smaller
    dimension to go to 0. It does so by center-cropping the
    larger dimension (consider resizing larger dimension?).

    Assume the image dimensions are l(arge), s(mall), then:
    s' = 384 -> l' = l * s' / s.
    Assume l' > 384 * 1333 / 800 (aka l / s > 1333 / 800), then:
    l'' = 384 * 1333 / 800 -> s'' = s' * l'' / l'.
    This means that:
    s'' = 384 * (384 * 1333 / 800) / (l * 384 / s)
    -> s'' = 384 * 1333 / 800 * s / l.
    Because the operation `//32` is performed on s'', we need:
    s'' >= 32 -> l / s <= 384 / 32 * 1333 / 800,
    otherwise:
    `ValueError: height and width must be > 0`.
    Note that this function triggers when l / s > 384 / 32 * 1333 / 800,
    which means our assumption of l' > 384 * 1333 / 800 is correct.
    """

    max_ratio = 384 / 32 * 1333 / 800

    def crop_largest(img):
        w, h = _get_wh(img)
        if max(w / h, h / w) > max_ratio:
            img = transforms.CenterCrop(
                (int(w * max_ratio), w) if h > w else (h, int(h * max_ratio))
            )(img)
        return img

    return transforms.Lambda(crop_largest)


def relative_random_crop(ratio=0.9):
    def transform(img):
        w, h = _get_wh(img)
        img = transforms.RandomCrop((int(ratio * h), int(ratio * w)))(img)
        return img

    return transforms.Lambda(transform)
