import base64
import io
import numpy as np
from PIL import Image, ImageChops
from src.settings import IMAGE_SIDE_PIXELS


def load_image_from_base64_uri(encoded_uri):
    encoded_uri = encoded_uri.split(',')[-1]
    decoded_bytes = base64.decodebytes(encoded_uri.encode())
    image = Image.open(io.BytesIO(decoded_bytes))
    background = Image.new('RGB', image.size, (255, 255, 255))
    background.paste(image, mask=image.split()[3])
    image = background.convert('RGB')
    image = image.convert('L')
    return image


def normalize_image(image, dpi=IMAGE_SIDE_PIXELS, binary_mode=True):
    image = 255 - image.reshape(dpi ** 2)
    if binary_mode:
        image[image != 0.0] = 0.99
        image[image == 0.0] = 0.01
    else:
        image = (image / 255.0 * 0.99) + 0.01
    return image


def trim_image(image):
    # source: https://stackoverflow.com/a/19271897
    bg = Image.new(image.mode, image.size, image.getpixel((0,0)))
    diff = ImageChops.difference(image, bg)
    diff = ImageChops.add(diff, diff, 2.0, -100)
    bbox = diff.getbbox()
    if bbox:
        return image.crop(bbox)


def resize_image(image, dpi=IMAGE_SIDE_PIXELS):
    return image.resize((dpi, dpi), Image.LANCZOS)


def crook_image(image, angle):
    image2 = image.convert('RGBA')
    image2 = image2.rotate(angle, resample=Image.BICUBIC)
    white = Image.new('RGBA', image2.size, (255,) * 4)
    image = Image.composite(image2, white, image2)
    image = image.convert('L')
    return image


def process_image(image):
    # needs image read by pillow!
    # trim, resize and normalize to original size
    image = trim_image(image)
    image = resize_image(image)
    # and now convert it to regular numpy array
    image = np.asarray(image, dtype=np.float32)
    image = normalize_image(image)
    return image


def open_image(image_filename):
    return Image.open(image_filename).convert('L')
