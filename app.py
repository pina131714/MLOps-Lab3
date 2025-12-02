import gradio as gr
import requests
import io
import os
from PIL import Image

RENDER_API_URL = "https://mlops-lab3-api.onrender.com"
PREDICT_ENDPOINT = f"{RENDER_API_URL}/predict"


def random_prediction_api(image: Image.Image):
    """
    Handles the Gradio image input, sends it to the Render FastAPI /predict endpoint,
    and returns the predicted class label.
    """
    if image is None:
        return "Please upload an image to predict its class."

    # 1. Convert the PIL Image object (received from Gradio) to a byte buffer
    img_byte_arr = io.BytesIO()
    # Save the image as JPEG into the buffer
    image.save(img_byte_arr, format='JPEG') 
    img_byte_arr.seek(0) # Rewind the buffer pointer to the start

    # 2. Prepare file payload for multipart/form-data upload
    # 'file' must match the parameter name in your FastAPI endpoint: file: UploadFile = File(...)
    files = {'file': ('image.jpg', img_byte_arr, 'image/jpeg')}

    try:
        # 3. Send POST request to the remote FastAPI API on Render
        response = requests.post(PREDICT_ENDPOINT, files=files, timeout=30)
        response.raise_for_status() # Raise exception for bad status codes (4xx or 5xx)
        
        data = response.json()
        
        # 4. Extract and display the prediction
        if 'prediction' in data:
            return f"Predicted Class: {data['prediction']}"
        
        return f"API Error: {data.get('detail', 'API response missing prediction.')}"

    except requests.exceptions.RequestException as e:
        # Handle connection errors, DNS errors, timeout, etc.
        return f"API Connection Error: Could not reach API or invalid response. Check Render URL and API status. ({e})"


# --- Gradio Interface ---

# Define the interface components
image_input = gr.Image(
    type="pil", 
    label="Input Image", 
    width=400
)
prediction_output = gr.Textbox(
    label="Random Prediction Result", 
    lines=1
)

# Build the Gradio interface
iface = gr.Interface(
    fn=random_prediction_api,
    inputs=image_input,
    outputs=prediction_output,
    title="MLOps Lab 3 Image Classifier GUI",
    description=f"A demo application using Gradio to send images to the containerized FastAPI backend (at {RENDER_API_URL}) for pet class predictions."
)

# Launch the GUI (necessary for local testing, ignored by HuggingFace Spaces)
if __name__ == "__main__":
    iface.launch()
