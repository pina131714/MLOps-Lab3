"""
Unit Testing of the application's logic (Image Processor)
"""
import io
import pytest
import numpy as np
from PIL import Image
from mylib.image_processor import (predict_image, resize_image, convert_to_grayscale, rotate_image, apply_blur,
								   normalize_image, get_image_info, CLASS_NAMES)

@pytest.fixture
def dummy_image():
    """Creates a simple 100x100 RGB red image for testing."""
    img = Image.new("RGB", (100, 100), color="red")
    return img

@pytest.fixture
def dummy_image_file(dummy_image):
    """Creates a file-like object (buffer) containing the dummy image."""
    buffer = io.BytesIO()
    dummy_image.save(buffer, format="JPEG")
    buffer.seek(0)
    return buffer

def test_predict_image(dummy_image):
    """Test that prediction returns a valid class name."""
    prediction = predict_image(dummy_image)
    assert isinstance(prediction, str)
    assert prediction in CLASS_NAMES

def test_resize_image(dummy_image_file):
    """Test that the image is resized to the correct dimensions."""
    # resize_image expects a file path or file-like object
    width, height = 50, 50
    resized_img = resize_image(dummy_image_file, width, height)
    
    assert resized_img.size == (width, height)

def test_convert_to_grayscale(dummy_image):
    """Test that the image is converted to grayscale (Mode 'L')."""
    gray_img = convert_to_grayscale(dummy_image)
    assert gray_img.mode == "L"

def test_rotate_image(dummy_image):
    """Test that the image rotates without errors."""
    angle = 90
    rotated_img = rotate_image(dummy_image, angle)
    
    # Rotating 90 degrees shouldn't change the size of a square image 
    # significantly in this context, but we mainly check it returns an Image
    assert isinstance(rotated_img, Image.Image)
    assert rotated_img.size is not None

def test_apply_blur(dummy_image):
    """Test that the blur filter runs and returns an image."""
    blurred_img = apply_blur(dummy_image, radius=2)
    assert isinstance(blurred_img, Image.Image)
    # Ideally, we could check pixel values changed, but ensuring it runs is enough for unit tests

def test_normalize_image(dummy_image):
    """Test that normalization returns a numpy array with values between 0 and 1."""
    norm_array = normalize_image(dummy_image)
    
    assert isinstance(norm_array, np.ndarray)
    # Pixel values should be floats
    assert norm_array.dtype == np.float64 or norm_array.dtype == np.float32
    # Values should be normalized
    assert norm_array.min() >= 0.0
    assert norm_array.max() <= 1.0

def test_get_image_info(dummy_image):
    """Test that metadata extraction returns correct information."""
    info = get_image_info(dummy_image)
    
    assert isinstance(info, dict)
    assert info["width"] == 100
    assert info["height"] == 100
    assert info["is_color"] is True
    assert "size" in info
