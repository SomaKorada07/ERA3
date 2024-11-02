# main.py
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
import numpy as np
import cv2
import librosa
import os
import soundfile as sf
import base64
import random

app = FastAPI()

# Mount the static directory
app.mount("/static", StaticFiles(directory="temp"), name="static")

# Global variable to store the loaded data
loaded_data = None
loaded_file_type = None
uploaded_audio_path = None
preprocessed_audio_path = None
augmented_audio_path = None
uploaded_image = None
preprocessed_image = None
augmented_image = None
uploaded_text = None
preprocessed_text = None
augmented_text = None

def encode_image(image):
    """Encode an image to a Base64 string."""
    _, buffer = cv2.imencode('.png', image)
    return base64.b64encode(buffer).decode('utf-8')

@app.get("/", response_class=HTMLResponse)
async def read_root():
    return """
    <html>
        <head>
            <title>File Upload</title>
            <style>
                body {
                    font-family: Arial, sans-serif;
                    margin: 0;
                    padding: 0;
                    background: linear-gradient(to right, #e0f7fa, #80deea); /* Gradient background */
                    display: flex;
                    justify-content: center; /* Center horizontally */
                    align-items: center; /* Center vertically */
                    height: 100vh; /* Full viewport height */
                }
                header {
                    background-color: #007bff;
                    color: white;
                    padding: 20px;
                    text-align: center;
                    box-shadow: 0 2px 5px rgba(0, 0, 0, 0.2);
                    position: absolute; /* Position header at the top */
                    top: 0;
                    width: 100%;
                }
                h1 {
                    margin: 0;
                    font-size: 24px;
                }
                .container {
                    max-width: 600px;
                    padding: 20px;
                    background: white;
                    border-radius: 8px;
                    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
                    z-index: 1; /* Ensure the container is above the header */
                }
                .form-group {
                    margin-bottom: 20px;
                }
                label {
                    display: block;
                    margin-bottom: 8px;
                    font-weight: bold;
                    color: #333;
                }
                input[type="file"] {
                    width: 100%;
                    padding: 10px;
                    border: 1px solid #ccc;
                    border-radius: 4px;
                    box-sizing: border-box;
                }
                .file-type {
                    margin: 15px 0;
                }
                .file-type label {
                    display: inline-block;
                    margin-right: 15px;
                    font-weight: normal;
                }
                input[type="radio"] {
                    margin-right: 5px;
                }
                input[type="submit"] {
                    background-color: #007bff;
                    color: white;
                    padding: 12px 15px;
                    border: none;
                    border-radius: 4px;
                    cursor: pointer;
                    width: 100%;
                    font-size: 16px;
                    transition: background-color 0.3s;
                }
                input[type="submit"]:hover {
                    background-color: #0056b3;
                }
                footer {
                    text-align: center;
                    margin-top: 20px;
                    font-size: 0.9em;
                    color: #666;
                }
            </style>
        </head>
        <body>
            <header>
                <h1>File Upload and Processing</h1>
            </header>
            <div class="container">
                <form action="/uploadfile/" method="post" enctype="multipart/form-data">
                    <div class="form-group">
                        <label for="file">Choose a file:</label>
                        <input type="file" name="file" id="file" required>
                    </div>
                    <div class="form-group file-type">
                        <label>File Type:</label>
                        <input type="radio" name="file_type" value="text" id="text" required>
                        <label for="text">Text File</label>
                        <input type="radio" name="file_type" value="image" id="image">
                        <label for="image">Image</label>
                        <input type="radio" name="file_type" value="audio" id="audio">
                        <label for="audio">Audio File</label>
                    </div>
                    <input type="submit" value="Upload">
                </form>
            </div>
        </body>
    </html>
    """

@app.post("/uploadfile/")
async def upload_file(file: UploadFile = File(...), file_type: str = Form(...)):
    global loaded_data, loaded_file_type, uploaded_audio_path, preprocessed_audio_path, augmented_audio_path
    global uploaded_image, preprocessed_image, augmented_image
    global uploaded_text, preprocessed_text, augmented_text

    # Reset paths for new upload
    uploaded_audio_path = None
    preprocessed_audio_path = None
    augmented_audio_path = None
    uploaded_image = None
    preprocessed_image = None
    augmented_image = None
    uploaded_text = None
    preprocessed_text = None
    augmented_text = None

    loaded_file_type = file_type
    file_location = f"temp/{file.filename}"
    with open(file_location, "wb+") as file_object:
        file_object.write(file.file.read())
    
    # Load the data based on file type
    if file_type == "image":
        loaded_data = cv2.imread(file_location)
        uploaded_image = encode_image(loaded_data)  # Encode image to Base64 for display
    elif file_type == "audio":
        loaded_data, _ = librosa.load(file_location, sr=None)
        uploaded_audio_path = file_location  # Store the original audio path
    elif file_type == "text":
        with open(file_location, 'r') as f:
            loaded_data = f.read()
            uploaded_text = loaded_data  # Store the uploaded text
            uploaded_audio_path = None  # No audio for text files
    else:
        return JSONResponse(content={"error": "Unsupported file type"}, status_code=400)

    return HTMLResponse(content=render_response())

@app.post("/preprocess/")
async def preprocess():
    global loaded_data, loaded_file_type, preprocessed_audio_path, preprocessed_image, preprocessed_text
    if loaded_file_type == "image":
        # Example preprocessing: Convert to grayscale
        loaded_data = cv2.cvtColor(loaded_data, cv2.COLOR_BGR2GRAY)
        preprocessed_image = encode_image(loaded_data)  # Encode preprocessed image to Base64
    elif loaded_file_type == "audio":
        # Example preprocessing: Normalize audio
        loaded_data = librosa.util.normalize(loaded_data)  # Normalize the audio
        preprocessed_audio_path = "temp/preprocessed_audio.wav"
        sf.write(preprocessed_audio_path, loaded_data, 22050)  # Save preprocessed audio
    elif loaded_file_type == "text":
        # Example preprocessing: Convert to lowercase
        preprocessed_text = uploaded_text.lower()  # Store preprocessed text
        preprocessed_image = None  # No image for text files
    else:
        return JSONResponse(content={"error": "Unsupported file type"}, status_code=400)

    return HTMLResponse(content=render_response())

@app.post("/augment/")
async def augment():
    global loaded_data, loaded_file_type, augmented_audio_path, augmented_image, augmented_text
    if loaded_file_type == "image":
        # Example augmentation: Horizontal flip
        loaded_data = cv2.flip(loaded_data, 1)
        augmented_image = encode_image(loaded_data)  # Encode augmented image to Base64
    elif loaded_file_type == "audio":
        # Example augmentation: Play audio at 0.5x speed
        loaded_data = librosa.effects.time_stretch(loaded_data, rate=0.5)  # Change rate to 0.5 for slower playback
        augmented_audio_path = "temp/augmented_audio.wav"
        sf.write(augmented_audio_path, loaded_data, 22050)  # Save augmented audio
    elif loaded_file_type == "text":
        # Example augmentation: Repeat a random word in the text
        words = uploaded_text.split()
        if words:  # Check if there are words to repeat
            word_to_repeat = random.choice(words)
            augmented_text = uploaded_text.replace(word_to_repeat, f"{word_to_repeat} {word_to_repeat}", 1)  # Repeat the word once
        else:
            augmented_text = uploaded_text  # No change if text is empty
        augmented_image = None  # No image for text files
    else:
        return JSONResponse(content={"error": "Unsupported file type"}, status_code=400)

    return HTMLResponse(content=render_response())

def render_response():
    audio_html_uploaded = ""
    audio_html_preprocessed = ""
    audio_html_augmented = ""

    # Handle uploaded audio
    if uploaded_audio_path:
        audio_html_uploaded += f'<h3>Original Audio</h3><audio controls><source src="/static/{os.path.basename(uploaded_audio_path)}" type="audio/wav">Your browser does not support the audio element.</audio>'
    
    # Handle preprocessed audio
    if preprocessed_audio_path:
        audio_html_preprocessed += f'<h3>Preprocessed Audio</h3><audio controls><source src="/static/{os.path.basename(preprocessed_audio_path)}" type="audio/wav">Your browser does not support the audio element.</audio>'
    
    # Handle augmented audio
    if augmented_audio_path:
        audio_html_augmented += f'<h3>Augmented Audio</h3><audio controls><source src="/static/{os.path.basename(augmented_audio_path)}" type="audio/wav">Your browser does not support the audio element.</audio>'

    return f"""
    <html>
        <head>
            <title>File Processing Results</title>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    margin: 0;
                    padding: 0;
                    background-color: #f4f4f4;
                }}
                header {{
                    background-color: #007bff;
                    color: white;
                    padding: 20px;
                    text-align: center;
                    box-shadow: 0 2px 5px rgba(0, 0, 0, 0.2);
                }}
                h1 {{
                    margin: 0;
                }}
                .section {{
                    margin: 20px auto;
                    padding: 15px;
                    background: white;
                    border-radius: 8px;
                    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
                    max-width: 500px;
                }}
                h2 {{
                    color: #333;
                    border-bottom: 2px solid #007bff;
                    padding-bottom: 10px;
                }}
                img {{
                    max-width: 100%;
                    height: auto;
                    border-radius: 4px;
                }}
                pre {{
                    background-color: #f8f9fa;
                    padding: 10px;
                    border-radius: 4px;
                    overflow: auto;
                }}
                .button-container {{
                    display: flex;
                    justify-content: space-between;
                    margin-top: 20px;
                }}
                .button-container input {{
                    flex: 1;
                    margin: 0 5px; /* Add some space between buttons */
                }}
                footer {{
                    text-align: center;
                    margin-top: 20px;
                    font-size: 0.9em;
                    color: #666;
                }}
            </style>
        </head>
        <body>
            <header>
                <h1>File Processing Results</h1>
            </header>
            <div class="section">
                <h2>Uploaded Content</h2>
                {audio_html_uploaded}
                <img src="data:image/png;base64,{uploaded_image}" alt="Uploaded Image" style="display: {'block' if uploaded_image else 'none'};"/>
                <pre style="display: {'block' if uploaded_text else 'none'};">{uploaded_text}</pre>
            </div>
            <div class="section">
                <h2>Preprocessed Content</h2>
                {audio_html_preprocessed}
                <img src="data:image/png;base64,{preprocessed_image}" alt="Preprocessed Image" style="display: {'block' if preprocessed_image else 'none'};"/>
                <pre style="display: {'block' if preprocessed_text else 'none'};">{preprocessed_text}</pre>
                <div class="button-container">
                    <form action="/preprocess/" method="post">
                        <input type="submit" value="Preprocess">
                    </form>
                </div>
            </div>
            <div class="section">
                <h2>Augmented Content</h2>
                {audio_html_augmented}
                <img src="data:image/png;base64,{augmented_image}" alt="Augmented Image" style="display: {'block' if augmented_image else 'none'};"/>
                <pre style="display: {'block' if augmented_text else 'none'};">{augmented_text}</pre>
                <div class="button-container">
                    <form action="/augment/" method="post">
                        <input type="submit" value="Augment">
                    </form>
                </div>
            </div>
            <footer>
                <a href="/">Upload another file</a>
            </footer>
        </body>
    </html>
    """

# Ensure to create a temp directory for file uploads
if not os.path.exists("temp"):
    os.makedirs("temp")