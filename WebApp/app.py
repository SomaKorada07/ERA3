from flask import Flask, render_template, request, jsonify, url_for
from werkzeug.utils import secure_filename
import os
import time
import random
import requests

app = Flask(__name__, static_url_path='/static')

# Configure upload folder
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')


def get_static_animal_image(animal):
    
    # Dictionary of placeholder image URLs for each animal
    image_urls = {
        'cat': [
            'https://placekitten.com/300/300'
        ],
        'dog': [
            'https://placedog.net/300/300',
            'https://placedog.net/301/301',
            'https://placedog.net/302/302'
        ],
        'elephant': [
            'https://files.worldwildlife.org/wwfcmsprod/images/African_Elephant_Kenya_112367/hero_small/3v49raxlb8_WW187785.jpg'
        ]
    }
    
    # Select a random image URL for the chosen animal
    image_url = random.choice(image_urls.get(animal, []))
    
    return image_url

@app.route('/upload_file', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    
    if file:
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        file_size = os.path.getsize(file_path)
        file_type = file.content_type
        
        return jsonify({
            'name': filename,
            'size': f"{file_size} bytes",
            'type': file_type
        })

def get_animal_image(animal):
    url = "https://api.starryai.com/creations/"

    payload = {
        "model": "lyra",
        "aspectRatio": "square",
        "highResolution": False,
        "images": 1,
        "steps": 20,
        "initialImageMode": "color",
        "prompt": f"a creative but natural looking image of {animal} in a forest."
    }
    headers = {
        "accept": "application/json",
        "content-type": "application/json",
        "X-API-Key": "JW7BOz2Ws_itvW9uz_2GxeobrgTxdg"
    }

    response = requests.post(url, json=payload, headers=headers)

    if response.status_code == 200:
        response_text = response.json()
        id = response_text['id']

        print(f"image generated with ID as {id}")

        time.sleep(10)

        image_url = get_image_url(id)

        if not image_url:
            image_url = get_static_animal_image(animal)
            print(f"static image url is {image_url}")

        return image_url


def get_image_url(id):

        id_url = f"https://api.starryai.com/creations/{id}"

        headers = {
            "accept": "application/json",
            "X-API-Key": "JW7BOz2Ws_itvW9uz_2GxeobrgTxdg"
        }

        image_url = None

        id_response = requests.get(id_url, headers=headers)

        id_response_text = id_response.json()

        image_url = id_response_text['images'][0]['url']

        if image_url:
            print(f"AI generated image url is {image_url}")

        return image_url


@app.route('/animal/<animal_name>')
def show_animal_image(animal_name):
    image_url = get_animal_image(animal_name)
    return jsonify({'image_url': image_url})  # Return JSON response

if __name__ == '__main__':
    app.run(debug=True)
