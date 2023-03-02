from flask import Flask, jsonify, request
from flask_cors import CORS
import pandas as pd
import os

app = Flask(__name__)
CORS(app)


def corsify_response(response):
    response.headers.add("Access-Control-Allow-Origin", "*")
    return response


@app.route('/getFileHeader', methods=['POST'])
def getFile():
    data = request.get_json()

    file_path = os.path.join('..', 'datasets', data['selected_file_path'])

    file_header = {"file_header": pd.read_csv(file_path, nrows=1).columns.tolist()}

    return corsify_response(jsonify(file_header))


if __name__ == '__main__':
    app.run(debug=True)
