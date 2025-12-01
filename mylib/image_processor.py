"""
Image Processing Library
"""

import random
import numpy as np
from PIL import Image, ImageFilter

# A list of class names for your "prediction"
CLASS_NAMES = ["cat", "dog", "bird", "tree", "bicycle"]


def predict_image(image: Image.Image) -> str: # pylint: disable=W0613
    """
    Predicts the class of a given image.

    NOTE: For Lab 1, this is a random choice from CLASS_NAMES.
    
    Args:
        image (Image.Image): The input PIL Image object (not used for this lab).

    Returns:
        str: A randomly chosen class name.
    """
    # The 'image' argument is a placeholder for now.
    # In future labs, you'll pass this to a real model.
    return random.choice(CLASS_NAMES)


def resize_image(image_file, width: int, height: int) -> Image.Image:
    """
    Resizes an image to a specific width and height.
    
    Args:
        image_file: A file-like object or path to the image.
        width (int): The target width.
        height (int): The target height.

    Returns:
        Image.Image: The resized PIL Image object.
    """
    with Image.open(image_file) as img:
        # Ensure image is in RGB mode for consistent array shape
        img = img.convert("RGB")
        resized_img = img.resize((width, height))
    return resized_img

# --- Optional Methods ---

def convert_to_grayscale(image: Image.Image) -> Image.Image:
    """
Optional: Converts a PIL Image to grayscale.
    
    Args:
        image (Image.Image): The input PIL Image object.

    Returns:
        Image.Image: The grayscale PIL Image object.
    """
    return image.convert("L")


def rotate_image(image: Image.Image, angle: float) -> Image.Image:
    """
Optional: Rotates a PIL Image by a given angle.
    
    Args:
        image (Image.Image): The input PIL Image object.
        angle (float): The angle (in degrees) to rotate the image.

    Returns:
        Image.Image: The rotated PIL Image object.
    """
    # 'expand=True' ensures the full rotated image is captured
    return image.rotate(angle, expand=True)


def apply_blur(image: Image.Image, radius: int = 2) -> Image.Image:
    """
    Optional: Applies a Gaussian blur filter to the image.
    
    Args:
        image (Image.Image): The input PIL Image object.
        radius (int): The radius of the blur. Defaults to 2.

    Returns:
        Image.Image: The blurred PIL Image object.
    """
    return image.filter(ImageFilter.GaussianBlur(radius=radius))


def normalize_image(image: Image.Image) -> np.ndarray:
    """
    Optional: Normalizes a PIL Image.
    
    Converts the image to a numpy array and scales pixel 
    values from [0, 255] to [0.0, 1.0].
    
    Args:
        image (Image.Image): The input PIL Image object.

    Returns:
        np.ndarray: A numpy array of the image with normalized pixel values.
    """
    # Ensure image is in RGB mode for consistent array shape
    img_rgb = image.convert("RGB")
    
    # Convert PIL Image to numpy array
    img_array = np.asarray(img_rgb)
    
    # Scale pixel values to [0.0, 1.0]
    normalized_array = img_array / 255.0
    
    return normalized_array


def get_image_info(image: Image.Image) -> dict:
    """
    Optional: Extracts metadata from a PIL Image.
    
    Args:
        image (Image.Image): The input PIL Image object.

    Returns:
        dict: A dictionary containing image metadata.
    """
    return {
        "size": image.size,  # (width, height)
        "width": image.width,
        "height": image.height,
        "mode": image.mode,  # 'L' (grayscale), 'RGB', 'RGBA'
        "format": image.format,  # 'JPEG', 'PNG'
        "channels": len(image.getbands()),
        "is_color": image.mode not in ('L', '1')
    }
