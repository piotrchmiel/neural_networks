import base64
import io
import os
import joblib
import numpy as np
from PIL import Image
from flask import Flask, render_template, request, send_from_directory
from src.utils import load_neural_network
from src.settings import OBJECT_DIR, IMAGE_SIDE_PIXELS


app = Flask(__name__, static_url_path='/static', static_folder='server/static', template_folder='server/templates')
uji = None
nn = None


@app.route('/')
def index():
    return render_template('index.html', image_side_pixels=IMAGE_SIDE_PIXELS)


@app.route('/static/<path:path>')
def send_static(path):
    return send_from_directory('static', path)


@app.route('/upload', methods=['POST'])
def process_image():
    data = request.form['image']
    image = load_image_from_base64_uri(data)
    image = normalize_image(image)
    return nn.predict(image, uji.map_result)


def load_image_from_base64_uri(encoded_uri):
    encoded_uri = encoded_uri.split(',')[-1]
    decoded_bytes = base64.decodebytes(encoded_uri.encode())
    image = Image.open(io.BytesIO(decoded_bytes))
    background = Image.new('RGB', image.size, (255, 255, 255))
    background.paste(image, mask=image.split()[3])
    image = background.convert('RGB')
    image = image.convert('F')
    image = np.array(image)
    return image


def normalize_image(image):
    image = 255 - image.reshape(IMAGE_SIDE_PIXELS ** 2)
    image = (image / 255.0 * 0.99) + 0.01
    return image


def init_server():
    global uji, nn
    uji = joblib.load(os.path.join(OBJECT_DIR, "Uji"))
    nn = load_neural_network()


if __name__ == '__main__':
    init_server()
    app.run()
