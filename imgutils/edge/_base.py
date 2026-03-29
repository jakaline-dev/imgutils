import math
from typing import Optional

import cv2
from PIL import Image

from ..data import ImageTyping, MultiImagesTyping, load_image, istack, normalize_multi_images, restore_multi_images_result


def cv2_resize(input_image, width, height):
    _origin_height, _origin_width, _ = input_image.shape
    k = max(width / _origin_width, height / _origin_height)
    return cv2.resize(
        input_image, (width, height),
        interpolation=cv2.INTER_LANCZOS4 if k > 1 else cv2.INTER_AREA
    )  # when channel is 1, the return value will be HxW instead of HxWxC


def resize_image(input_image, resolution, align: int = 64):
    height, width, _ = input_image.shape
    k = float(resolution) / min(height, width)
    new_height, new_width = height * k, width * k
    if align:
        new_height = int(math.ceil(height / align)) * align
        new_width = int(math.ceil(width / align)) * align

    return cv2_resize(input_image, new_width, new_height)


def _get_image_edge(image: MultiImagesTyping, edge_func, backcolor: str = 'white', forecolor: Optional[str] = None):
    images, is_multi = normalize_multi_images(image)
    results = []
    for image_item in images:
        image_item = load_image(image_item, mode='RGB')
        edge = edge_func(image_item)

        is_transparent = backcolor.lower() == 'transparent'
        back_image = Image.new('RGBA', (image_item.width, image_item.height), '#ffffff' if is_transparent else backcolor)
        back_alpha = 0.0 if is_transparent else 1.0
        fore_image = image_item if forecolor is None else Image.new('RGBA', (image_item.width, image_item.height), forecolor)

        retval = istack((back_image, back_alpha), (fore_image, edge))
        results.append(retval if is_transparent else retval.convert('RGB'))
    return restore_multi_images_result(results, is_multi)
