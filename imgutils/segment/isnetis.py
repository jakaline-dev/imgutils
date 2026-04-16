"""
Overview:
    Anime character segmentation, based on https://huggingface.co/skytnt/anime-seg .
"""

import cv2
import numpy as np
from ..utils.hf import hf_hub_download as _hf_hub_download

from ..data import (
    ImageTyping, MultiImagesTyping, load_image, istack,
    normalize_multi_images, restore_multi_images_result,
)
from ..utils import ts_lru_cache
from ..utils.onnxruntime import open_onnx_model


@ts_lru_cache()
def _get_model():
    return open_onnx_model(_hf_hub_download("skytnt/anime-seg", "isnetis.onnx"))


def _get_isnetis_mask_single(image: ImageTyping, scale: int = 1024):
    """
    Overview:
        Get mask with isnetis.

    :param image: Original image (assume its size is ``(H, W)``).
    :param scale: Scale when passing it into neural network. Default is ``1024``,
        inspired by https://huggingface.co/spaces/skytnt/anime-remove-background/blob/main/app.py#L8 .
    :return: Get a mask with all the pixels, which shape is ``(H, W)``.
    """
    image = np.asarray(load_image(image, mode='RGB'))
    image = (image / 255).astype(np.float32)
    h, w = h0, w0 = image.shape[:-1]
    h, w = (scale, int(scale * w / h)) if h > w else (int(scale * h / w), scale)
    ph, pw = scale - h, scale - w
    img_input = np.zeros([scale, scale, 3], dtype=np.float32)
    img_input[ph // 2:ph // 2 + h, pw // 2:pw // 2 + w] = cv2.resize(image, (w, h))
    img_input = np.transpose(img_input, (2, 0, 1))
    img_input = img_input[np.newaxis, :]
    mask = _get_model().run(None, {'img': img_input})[0][0]
    mask = np.transpose(mask, (1, 2, 0))
    mask = mask[ph // 2:ph // 2 + h, pw // 2:pw // 2 + w]
    mask = cv2.resize(mask, (w0, h0))[:, :, np.newaxis]
    return mask.reshape(*mask.shape[:-1])


def get_isnetis_mask(image: MultiImagesTyping, scale: int = 1024):
    images, is_multi = normalize_multi_images(image)
    results = [_get_isnetis_mask_single(item, scale) for item in images]
    return restore_multi_images_result(results, is_multi)


def segment_with_isnetis(image: MultiImagesTyping, background: str = 'lime', scale: int = 1024):
    """
    Overview:
        Segment image with pure color background.

    :param image: Original image (assume its size is ``(H, W)``).
    :param background: Background color for padding. Default is ``lime`` which represents ``#00ff00``.
    :param scale: Scale when passing it into neural network. Default is ``1024``,
        inspired by https://huggingface.co/spaces/skytnt/anime-remove-background/blob/main/app.py#L8 .
    :return: The mask and An RGB image featuring a pure-colored background along with a segmented image.

    Examples::
        >>> from imgutils.segment import segment_with_isnetis
        >>>
        >>> mask_, image_ = segment_with_isnetis('hutao.png')
        >>> image_.save('hutao_seg.png')
        >>>
        >>> mask_, image_ = segment_with_isnetis('skadi.jpg', background='white')  # white background
        >>> image_.save('skadi_seg.jpg')

        The result should be

        .. image:: isnetis_color.plot.py.svg
           :align: center

    """
    images, is_multi = normalize_multi_images(image)
    results = []
    for image_item in images:
        image_item = load_image(image_item, mode='RGB')
        mask = _get_isnetis_mask_single(image_item, scale)
        results.append((mask, istack((background, 1.0), (image_item, mask)).convert('RGB')))
    return restore_multi_images_result(results, is_multi)


def segment_rgba_with_isnetis(image: MultiImagesTyping, scale: int = 1024):
    """
    Overview:
        Segment image with transparent background.

    :param image: Original image (assume its size is ``(H, W)``).
    :param scale: Scale when passing it into neural network. Default is ``1024``,
        inspired by https://huggingface.co/spaces/skytnt/anime-remove-background/blob/main/app.py#L8 .
    :return: The mask and An RGBA image featuring a transparent background along with a segmented image.

    Examples::
        >>> from imgutils.segment import segment_rgba_with_isnetis
        >>>
        >>> mask_, image_ = segment_rgba_with_isnetis('hutao.png')
        >>> image_.save('hutao_seg.png')
        >>>
        >>> mask_, image_ = segment_rgba_with_isnetis('skadi.jpg')
        >>> image_.save('skadi_seg.png')

        The result should be

        .. image:: isnetis_trans.plot.py.svg
           :align: center

    """
    images, is_multi = normalize_multi_images(image)
    results = []
    for image_item in images:
        image_item = load_image(image_item, mode='RGB')
        mask = _get_isnetis_mask_single(image_item, scale)
        results.append((mask, istack((image_item, mask))))
    return restore_multi_images_result(results, is_multi)
