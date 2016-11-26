from flask import Flask, render_template, send_from_directory

app = Flask(__name__, static_url_path='/static', static_folder='server/static', template_folder='server/templates')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/static/<path:path>')
def send_static(path):
    return send_from_directory('static', path)

if __name__ == '__main__':
    app.run()
