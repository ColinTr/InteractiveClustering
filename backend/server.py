from flask import Flask, jsonify, request, session
from flask_cors import CORS
import pandas as pd
import os

app = Flask(__name__)
app.secret_key = "My Secret key"
app.config['SESSION_TYPE'] = 'filesystem'
CORS(app)


def corsify_response(response):
    response.headers.add("Access-Control-Allow-Origin", "*")
    return response


@app.route('/getFileHeader', methods=['POST'])
def getFileHeader():
    data = request.get_json()
    file_path = os.path.join('..', 'datasets', data['selected_file_path'])
    dataset = pd.read_csv(file_path)
    session['loaded_dataset'] = dataset.to_json()
    columns_names = dataset.columns

    return corsify_response(jsonify({"file_header": columns_names.tolist()}))


@app.route('/getFeatureUniqueValues', methods=['POST'])
def getFeatureUniqueValues():
    dataset = session.get('loaded_dataset')

    if dataset:
        dataset = pd.DataFrame(eval(dataset))
        data = request.get_json()
        feature_name = data['feature_name']
        unique_values = dataset[feature_name].unique()
        return corsify_response(jsonify({"unique_values": unique_values.tolist()}))
    else:
        return "Dataset not loaded", 400


if __name__ == '__main__':
    app.run(debug=True)
