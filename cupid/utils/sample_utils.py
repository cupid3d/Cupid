from PIL import Image
from typing import *

def pad_to_square(image: Image.Image) -> Image.Image:
    W, H = image.size
    if W == H:
        return image

    max_size = max(W, H)
    if image.mode == "RGBA":
        fill_value = (0, 0, 0, 0)
    elif image.mode == "RGB":
        fill_value = (0, 0, 0)
    elif image.mode == "L":
        fill_value = 0
    else:
        raise ValueError(f"Unsupported image mode: {image.mode}")
    
    padded_image = Image.new(image.mode, (max_size, max_size), fill_value)
    padded_image.paste(image, ((max_size - W) // 2, (max_size - H) // 2))
    # bbox = ((max_size - W) // 2, (max_size - H) // 2, (max_size - W) // 2 + W, (max_size - H) // 2 + H)
    return padded_image

def load_image(image_path: str, pad_square=True) -> Image.Image:
    input_image = Image.open(image_path)
    if pad_square:
        input_image = pad_to_square(input_image)
    return input_image