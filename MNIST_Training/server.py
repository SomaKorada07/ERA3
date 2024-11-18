from flask import Flask, render_template, jsonify
import os

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/training_history')
def training_history():
    if os.path.exists('static/training_history.json'):
        with open('static/training_history.json', 'r') as f:
            return jsonify(f.read())
    return jsonify([])

@app.route('/test_samples')
def test_samples():
    if os.path.exists('static/test_samples.json'):
        with open('static/test_samples.json', 'r') as f:
            return jsonify(f.read())
    return jsonify([])

if __name__ == '__main__':
    app.run(debug=True) 