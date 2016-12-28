import os
import joblib
import json
from flask import Flask, render_template, request, send_from_directory, abort
from src.utils import load_neural_network
from src.image_utils import load_image_from_base64_uri, process_image
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
def handle_image():
    data = request.form['image']
    image = load_image_from_base64_uri(data)
    try:
        image = process_image(image)
        return json.dumps(nn.get_all_predictions(image, dataset.map_result))
    except:
        abort(500)

@app.route('/save_character', methods=['POST'])
def handle_character():
    character = request.form['character']
    image = load_image_from_base64_uri(request.form['image'])
    try:
        return json.dumps({'success':True}), 200, {'ContentType': 'application/json'}
    except:
        abort(500)

def init_server():
    global dataset, nn
    dataset = joblib.load(os.path.join(OBJECT_DIR, "Uji"))
    # dataset = joblib.load(os.path.join(OBJECT_DIR, "Mnist"))
    nn = load_neural_network()

if __name__ == '__main__':
    init_server()
    app.run()
