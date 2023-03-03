from flask import Flask, jsonify, request, session, send_file
from matplotlib.figure import Figure
from sklearn.manifold import TSNE
from flask_cors import CORS
import seaborn as sns
import pandas as pd
import numpy as np
import datetime
import json
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


def saveResultsDict(results_dict):
    with open(os.path.join('.', 'results', 'results_dict.json'), 'w') as fp:
        json.dump(results_dict, fp)


def loadResultsDict():
    if not os.path.isfile(os.path.join('.', 'results', 'results_dict.json')):
        saveResultsDict({})

    with open(os.path.join('.', 'results', 'results_dict.json'), 'r') as fp:
        results_dict = json.load(fp)

    return results_dict


@app.route('/getDatasetTSNE', methods=['POST'])
def getDatasetTSNE():
    dataset = session.get('loaded_dataset')
    if dataset:
        dataset = pd.DataFrame(eval(dataset))
        data = request.get_json()

        dataset_name = data['dataset_name']
        selected_features = data['selected_features']
        tsne_seed = data['tsne_seed']
        tsne_perplexity = data['tsne_perplexity']
        known_classes = data['known_classes']
        unknown_classes = data['unknown_classes']
        color_by = data['color_by']
        target_name = data['target_name']

        tsne_array = None

        # 1) Try to find the configuration in the results_dict
        results_dict = loadResultsDict()
        corresponding_tsne_config_name = None
        if dataset_name in results_dict.keys():
            for tsne_config_name in results_dict[dataset_name].keys():
                tsne_config = results_dict[dataset_name][tsne_config_name]
                if tsne_config['target_name'] == target_name and \
                        set(tsne_config["used_features"]) == set(selected_features) and \
                        tsne_config["tsne_seed"] == tsne_seed and \
                        tsne_config["tsne_perplexity"] == tsne_perplexity:
                    if os.path.isfile(tsne_config['tsne_array_filepath']):
                        tsne_array = pd.read_csv(tsne_config['tsne_array_filepath'], header=None)
                        corresponding_tsne_config_name = tsne_config_name
                        break

        # 2) If this configuration wasn't found in the configuration, it needs to be run
        if tsne_array is None:
            tsne_array = TSNE(n_components=2, perplexity=tsne_perplexity, random_state=tsne_seed).fit_transform(dataset[selected_features])
            tsne_array = pd.DataFrame(tsne_array)

            # And it is then saved
            tsne_array_folder_path = os.path.join('.', 'results', 'tsne_arrays', dataset_name)

            if not os.path.isdir(tsne_array_folder_path):
                os.makedirs(tsne_array_folder_path)

            tsne_datetime_string = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

            tsne_array_filename = tsne_datetime_string + '.csv'

            tsne_array.to_csv(path_or_buf=os.path.join(tsne_array_folder_path, tsne_array_filename), index=False, header=None)

            if dataset_name not in results_dict.keys():
                results_dict[dataset_name] = {}

            results_dict[dataset_name]["tsne_config_" + tsne_datetime_string] = {
                "used_features": selected_features,
                "tsne_seed": tsne_seed,
                "tsne_perplexity": tsne_perplexity,
                "target_name": target_name,
                "tsne_array_filepath": os.path.join(tsne_array_folder_path, tsne_array_filename),
                "images_config": {}
            }

            saveResultsDict(results_dict)
            corresponding_tsne_config_name = "tsne_config_" + tsne_datetime_string

        # 3) Try to find if the image was already generated beforehand
        image_filepath = None
        tsne_config = results_dict[dataset_name][corresponding_tsne_config_name]
        for image_config_name in tsne_config['images_config'].keys():
            image_config = tsne_config['images_config'][image_config_name]
            if set(image_config['known_classes']) == set(known_classes) and set(image_config['unknown_classes']) == set(unknown_classes):
                if color_by == 'known_only':
                    if image_config['color_by'] == 'known_only':
                        # This is the same image, send it back to React
                        if os.path.isfile(image_config['image_filepath']):
                            image_filepath = image_config['image_filepath']
                # ToDo : support for other types of coloring here...

        if image_filepath is not None:
            return send_file(image_filepath, mimetype='image/png')
        else:
            # If the image doesn't exist, we need to create it
            image_folder_path = os.path.join('.', 'results', 'images_folder', dataset_name)

            if not os.path.isdir(image_folder_path):
                os.makedirs(image_folder_path)

            image_datetime_string = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

            image_filename = image_datetime_string + '.png'

            fig = Figure()
            axis = fig.add_subplot(1, 1, 1)
            target_array = np.array(dataset[target_name])
            target_array = [target if target in known_classes else "Unknown" for target in target_array]
            sns.scatterplot(ax=axis, x=np.array(tsne_array[tsne_array.columns[0]]), y=np.array(tsne_array[tsne_array.columns[1]]), hue=target_array)
            fig.savefig(os.path.join(image_folder_path, image_filename), dpi=fig.dpi, bbox_inches='tight')

            results_dict[dataset_name][corresponding_tsne_config_name]['images_config']['image_' + image_datetime_string] = {
                "image_filepath": os.path.join(image_folder_path, image_filename),
                "known_classes": known_classes,
                "unknown_classes": unknown_classes,
                "color_by": color_by,
                "model_config": ""
            }

            saveResultsDict(results_dict)

            return send_file(os.path.join(image_folder_path, image_filename), mimetype='image/png')
    else:
        return "Dataset not loaded", 400


if __name__ == '__main__':
    app.run(debug=True)
