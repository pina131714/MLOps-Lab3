"""
Integration testing with the API
"""
import io
import pytest
from PIL import Image
from fastapi.testclient import TestClient
from api.api import app

# --- Fixtures ---

@pytest.fixture
def client():
    """Testing client from FastAPI."""
    return TestClient(app)

@pytest.fixture
def dummy_image_bytes():
    """
    Creates a simple 100x100 RGB red image in memory.
    Returns the bytes, ready to be sent as a file.
    """
    img = Image.new("RGB", (100, 100), color="red")
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='JPEG')
    img_byte_arr.seek(0)  # Reset pointer to the beginning
    return img_byte_arr

# --- Tests ---

def test_home_endpoint(client):
    """Verify that the home endpoint returns 200 OK."""
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]

def test_predict_endpoint(client, dummy_image_bytes):
    """Verify that /predict returns a prediction."""
    # We simulate a file upload. 
    # Key "file" matches the name in the API function: file: UploadFile = File(...)
    files = {"file": ("test_image.jpg", dummy_image_bytes, "image/jpeg")}
    
    response = client.post("/predict", files=files)
    
    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
    assert isinstance(data["prediction"], str)

def test_resize_endpoint(client, dummy_image_bytes):
    """Verify that /resize returns an image with the correct dimensions."""
    files = {"file": ("test_image.jpg", dummy_image_bytes, "image/jpeg")}
    # Form data must be sent as a dictionary
    data = {"width": "50", "height": "50"}
    
    response = client.post("/resize", files=files, data=data)
    
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/jpeg"
    
    # Verify the output image size
    resized_image = Image.open(io.BytesIO(response.content))
    assert resized_image.size == (50, 50)

def test_info_endpoint(client, dummy_image_bytes):
    """Verify that /info returns correct metadata."""
    files = {"file": ("test_image.jpg", dummy_image_bytes, "image/jpeg")}
    
    response = client.post("/info", files=files)
    
    assert response.status_code == 200
    data = response.json()
    assert data["width"] == 100
    assert data["height"] == 100
    assert data["format"] == "JPEG"

def test_grayscale_endpoint(client, dummy_image_bytes):
    """Verify that /grayscale returns a PNG image."""
    files = {"file": ("test_image.jpg", dummy_image_bytes, "image/jpeg")}
    
    response = client.post("/grayscale", files=files)
    
    assert response.status_code == 200
    # Our API converts grayscale to PNG
    assert response.headers["content-type"] == "image/png"

def test_rotate_endpoint(client, dummy_image_bytes):
    """Verify that /rotate works with an angle."""
    files = {"file": ("test_image.jpg", dummy_image_bytes, "image/jpeg")}
    data = {"angle": "90"}
    
    response = client.post("/rotate", files=files, data=data)
    
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"

def test_blur_endpoint(client, dummy_image_bytes):
    """Verify that /blur works with a radius."""
    files = {"file": ("test_image.jpg", dummy_image_bytes, "image/jpeg")}
    data = {"radius": "5"}
    
    response = client.post("/blur", files=files, data=data)
    
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/jpeg"

def test_normalize_endpoint(client, dummy_image_bytes):
    """Verify that /normalize returns statistics."""
    files = {"file": ("test_image.jpg", dummy_image_bytes, "image/jpeg")}
    
    response = client.post("/normalize", files=files)
    
    assert response.status_code == 200
    data = response.json()
    assert "mean" in data
    assert "shape" in data

# --- Negative / Error Tests ---

def test_resize_missing_parameters(client, dummy_image_bytes):
    """Verify validation error when form data is missing."""
    files = {"file": ("test_image.jpg", dummy_image_bytes, "image/jpeg")}
    # Missing 'width' and 'height'
    
    response = client.post("/resize", files=files)
    
    # FastAPI returns 422 Unprocessable Entity for missing required form fields
    assert response.status_code == 422

def test_invalid_image_file(client):
    """Verify error handling for invalid image files."""
    # Sending a text file instead of an image
    files = {"file": ("test.txt", b"this is not an image", "text/plain")}
    
    response = client.post("/predict", files=files)
    
    # Depending on your API implementation, this might be 400 or 500.
    # In our api.py we catch exceptions and return 500 or 400 for bad images.
    assert response.status_code in [400, 500]
