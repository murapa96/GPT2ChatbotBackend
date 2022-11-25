from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import model

Model = model.Model
app = Flask(__name__)
CORS(app)
model = Model()


@app.route('/generate', methods=['GET'])
def generate():
    prompt = request.args.get('prompt')
    length = int(request.args.get('length'))

    return model.predict(prompt, length)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=9000)
