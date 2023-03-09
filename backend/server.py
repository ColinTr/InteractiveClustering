"""
Orange Labs
Authors : Colin Troisemaine
Maintainer : colin.troisemaine@gmail.com
"""

from ProjectionInClassifier import ProjectionInClassifier, NewThreadedTrainingTask
from sklearn.cluster import KMeans, SpectralClustering
from flask import Flask, jsonify, request, send_file
from sklearn.multiclass import OneVsRestClassifier
from sklearn.tree import DecisionTreeClassifier
from matplotlib.figure import Figure
from sklearn.manifold import TSNE
from PyPDF2 import PdfMerger
from flask_cors import CORS
from sklearn import tree
import seaborn as sns
import pandas as pd
import numpy as np
import datetime
import graphviz
import logging
import shutil
import json
import os

app = Flask(__name__)
logging.basicConfig(level=logging.DEBUG)
# app.secret_key = "My Secret key"
# app.config['SESSION_TYPE'] = 'filesystem'
# app.config.from_object(__name__)
# Session(app)
CORS(app)

# Global dict for storing data in between requests
# /!\ DO NOT USE IF THE SERVER IS DEPLOYED IN PRODUCTION /!\
session = {}
running_threads = {}


def corsify_response(response):
    response.headers.add("Access-Control-Allow-Origin", "*")
    return response


def equal_dicts(d1, d2, ignore_keys):
    d1_filtered = {k: v for k, v in d1.items() if k not in ignore_keys}
    d2_filtered = {k: v for k, v in d2.items() if k not in ignore_keys}
    return d1_filtered == d2_filtered


@app.route('/getFileHeader', methods=['POST'])
def getFileHeader():
    data = request.get_json()
    file_path = os.path.join('..', 'datasets', data['selected_file_path'])
    dataset = pd.read_csv(file_path)
    session['loaded_dataset'] = dataset  # dataset.to_json()
    columns_names = dataset.columns

    return corsify_response(jsonify({"file_header": columns_names.tolist()}))


@app.route('/getFeatureUniqueValues', methods=['POST'])
def getFeatureUniqueValues():
    if "loaded_dataset" not in session.keys():
        return jsonify({"error_message": "Dataset not loaded"}), 422

    dataset = session.get('loaded_dataset')
    data = request.get_json()
    feature_name = data['feature_name']
    unique_values = dataset[feature_name].unique()
    return corsify_response(jsonify({"unique_values": unique_values.tolist()})), 200


def saveResultsDict(results_dict):
    with open(os.path.join('.', 'results', 'results_dict.json'), 'w') as fp:
        json.dump(results_dict, fp)


def loadResultsDict():
    if not os.path.isfile(os.path.join('.', 'results', 'results_dict.json')):
        saveResultsDict({})

    with open(os.path.join('.', 'results', 'results_dict.json'), 'r') as fp:
        results_dict = json.load(fp)

    return results_dict


def runTSNE(results_dict, dataset_name, dataset, target_name, selected_features, tsne_seed, tsne_perplexity, known_classes, unknown_classes, show_unknown_only):
    if show_unknown_only is True:
        mask = np.in1d(np.array(dataset[target_name]), unknown_classes)
    else:
        mask = np.repeat(True, len(dataset))

    tsne_data = np.array(dataset[selected_features])[mask]

    tsne_array = TSNE(n_components=2, perplexity=tsne_perplexity, random_state=tsne_seed).fit_transform(tsne_data)
    tsne_array = pd.DataFrame(tsne_array)

    # And it is then saved
    tsne_array_folder_path = os.path.join('.', 'results', 'tsne_arrays', dataset_name)

    if not os.path.isdir(tsne_array_folder_path):
        os.makedirs(tsne_array_folder_path)

    tsne_datetime_string = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")

    tsne_array_filename = tsne_datetime_string + '.csv'

    tsne_array.to_csv(path_or_buf=os.path.join(tsne_array_folder_path, tsne_array_filename), index=False, header=None)

    if dataset_name not in results_dict.keys():
        results_dict[dataset_name] = {}

    results_dict[dataset_name]["tsne_config_" + tsne_datetime_string] = {
        "tsne_config": {
            "selected_features": selected_features,
            "known_classes": known_classes,
            "unknown_classes": unknown_classes,
            "target_name": target_name,
            "show_unknown_only": show_unknown_only,
            "tsne_seed": tsne_seed,
            "tsne_perplexity": tsne_perplexity
        },
        "tsne_filepath": os.path.join(tsne_array_folder_path, tsne_array_filename),
        "images_configurations": {}
    }

    saveResultsDict(results_dict)
    corresponding_tsne_config_name = "tsne_config_" + tsne_datetime_string

    return tsne_array, corresponding_tsne_config_name


def findTSNEConfig(results_dict, dataset_name, tsne_config_to_find):
    if dataset_name in results_dict.keys():
        for tsne_run_name in results_dict[dataset_name].keys():
            tsne_run = results_dict[dataset_name][tsne_run_name]

            # In the case where we show all classes, the known and unknown classes don't matter in the T-SNE
            if tsne_config_to_find['show_unknown_only'] is False and tsne_run["tsne_config"]['show_unknown_only'] is False:
                if equal_dicts(tsne_run["tsne_config"], tsne_config_to_find, ['known_classes', 'unknown_classes']):
                    if os.path.isfile(tsne_run['tsne_filepath']):
                        tsne_array = pd.read_csv(tsne_run['tsne_filepath'], header=None)
                        return tsne_array, tsne_run_name
            # If we show only the known classes, the whole config should be equal
            elif tsne_run["tsne_config"] == tsne_config_to_find:
                if os.path.isfile(tsne_run['tsne_filepath']):
                    tsne_array = pd.read_csv(tsne_run['tsne_filepath'], header=None)
                    return tsne_array, tsne_run_name

    return None, None


def findImage(results_dict, dataset_name, corresponding_tsne_config_name, image_configuration):
    tsne_config = results_dict[dataset_name][corresponding_tsne_config_name]
    for image_config_name in tsne_config['images_configurations'].keys():
        image_config = tsne_config['images_configurations'][image_config_name]

        if image_config['image_configuration'] == image_configuration:
            if os.path.isfile(image_config['image_filepath']):
                return image_config['image_filepath']

    return None


@app.route('/getDatasetTSNE', methods=['POST'])
def getDatasetTSNE():
    if "loaded_dataset" not in session.keys():
        return jsonify({"error_message": "Dataset not loaded"}), 422

    dataset = session.get('loaded_dataset')
    data = request.get_json()

    dataset_name = data['dataset_name']

    tsne_config = data['tsne_config']
    selected_features = tsne_config['selected_features']
    known_classes = tsne_config['known_classes']
    unknown_classes = tsne_config['unknown_classes']
    target_name = tsne_config['target_name']
    show_unknown_only = tsne_config['show_unknown_only']
    tsne_seed = tsne_config['tsne_seed']
    tsne_perplexity = tsne_config['tsne_perplexity']

    image_config = data['image_config']
    color_by = image_config['color_by']
    random_state = image_config['random_state']

    if show_unknown_only is True:
        unknown_mask = np.in1d(np.array(dataset[target_name]), unknown_classes)
    else:
        unknown_mask = np.repeat(True, len(dataset))
    tsne_target = np.array(dataset[target_name])[unknown_mask]
    tsne_target = [target if target in known_classes else "Unknown" for target in tsne_target]

    results_dict = loadResultsDict()

    # Try to find the configuration in the results_dict
    tsne_array, corresponding_tsne_config_name = findTSNEConfig(results_dict, dataset_name, tsne_config)

    if tsne_array is not None:
        # Try to find if the image was already generated beforehand
        image_filepath = findImage(results_dict, dataset_name, corresponding_tsne_config_name, image_config)
        if image_filepath is not None:
            return send_file(image_filepath, mimetype='image/png')

    # If this configuration wasn't found in the configuration, it needs to be run
    if tsne_array is None:
        tsne_array, corresponding_tsne_config_name = runTSNE(results_dict, dataset_name, dataset, target_name, selected_features, tsne_seed, tsne_perplexity, known_classes, unknown_classes, show_unknown_only)

    # If the image doesn't exist, we need to create it
    image_folder_path = os.path.join('.', 'results', 'images_folder', dataset_name)

    if not os.path.isdir(image_folder_path):
        os.makedirs(image_folder_path)

    image_datetime_string = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")

    image_filename = image_datetime_string + '.png'

    fig = Figure(figsize=(8, 8))
    axis = fig.add_subplot(1, 1, 1)
    axis.set_title("T-SNE of the original " + dataset_name + " dataset")
    sns.scatterplot(ax=axis, x=np.array(tsne_array[tsne_array.columns[0]]), y=np.array(tsne_array[tsne_array.columns[1]]), hue=tsne_target)

    fig.savefig(os.path.join(image_folder_path, image_filename), dpi=fig.dpi, bbox_inches='tight')

    results_dict[dataset_name][corresponding_tsne_config_name]['images_configurations']['image_' + image_datetime_string] = {
        "image_configuration": {
            "random_state": random_state,
            "color_by": color_by,
            "model_config": "",
            "known_classes": known_classes,
            "unknown_classes": unknown_classes
        },
        "image_filepath": os.path.join(image_folder_path, image_filename)
    }

    saveResultsDict(results_dict)

    return send_file(os.path.join(image_folder_path, image_filename), mimetype='image/png')


@app.route('/runClustering', methods=['POST'])
def runClustering():
    if "loaded_dataset" not in session.keys():
        return jsonify({"error_message": "Dataset not loaded"}), 422

    dataset = session.get('loaded_dataset')
    data = request.get_json()

    dataset_name = data['dataset_name']

    tsne_config = data['tsne_config']
    selected_features = tsne_config['selected_features']
    known_classes = tsne_config['known_classes']
    unknown_classes = tsne_config['unknown_classes']
    target_name = tsne_config['target_name']
    show_unknown_only = tsne_config['show_unknown_only']
    tsne_seed = tsne_config['tsne_seed']
    tsne_perplexity = tsne_config['tsne_perplexity']

    image_config = data['image_config']
    color_by = image_config['color_by']
    random_state = image_config['random_state']
    model_config = image_config['model_config']
    model_name = model_config['model_name']

    filtered_dataset = np.array(dataset[selected_features])

    results_dict = loadResultsDict()

    # Try to find the configuration in the results_dict
    tsne_array, corresponding_tsne_config_name = findTSNEConfig(results_dict, dataset_name, tsne_config)

    # Try to find if the image was already generated beforehand
    if tsne_array is not None:
        image_filepath = findImage(results_dict, dataset_name, corresponding_tsne_config_name, image_config)
        if image_filepath is not None:
            return send_file(image_filepath, mimetype='image/png')

    # If this configuration wasn't found in the configuration, it needs to be run
    if tsne_array is None:
        tsne_array, corresponding_tsne_config_name = runTSNE(results_dict, dataset_name, dataset, target_name, selected_features, tsne_seed, tsne_perplexity, known_classes, unknown_classes, show_unknown_only)

    # Generate the image based on the prediction
    if model_name == "k_means":
        k_means_n_clusters = model_config['k_means_n_clusters']

        # We train only on the unknown data
        unknown_mask = np.in1d(np.array(dataset[target_name]), unknown_classes)

        kmeans_model = KMeans(n_clusters=k_means_n_clusters,
                              random_state=random_state,
                              n_init="auto")
        clustering_prediction = kmeans_model.fit_predict(filtered_dataset[unknown_mask])

        full_target_to_plot = np.array(dataset[target_name])
        full_target_to_plot[unknown_mask] = ["Cluster " + str(pred) for pred in clustering_prediction]

    elif model_name == "spectral_clustering":
        spectral_clustering_n_clusters = model_config['spectral_clustering_n_clusters']
        spectral_clustering_affinity = model_config['spectral_clustering_affinity']

        # We train only on the unknown data
        unknown_mask = np.in1d(np.array(dataset[target_name]), unknown_classes)
        spectral_clustering_model = SpectralClustering(n_clusters=spectral_clustering_n_clusters,
                                                       affinity=spectral_clustering_affinity,
                                                       random_state=random_state,
                                                       assign_labels='discretize')
        clustering_prediction = spectral_clustering_model.fit(filtered_dataset[unknown_mask])
        clustering_prediction = clustering_prediction.labels_

        full_target_to_plot = np.array(dataset[target_name])
        full_target_to_plot[unknown_mask] = ["Cluster " + str(pred) for pred in clustering_prediction]

    elif model_name == "tabularncd":
        print("ToDo tabularncd")
        tabncd_n_clusters = model_config["tabncd_n_clusters"]
        tabncd_cosine_topk = model_config["tabncd_cosine_topk"]
        tabncd_w1 = model_config["tabncd_w1"]
        tabncd_w2 = model_config["tabncd_w2"]
        tabncd_classifier_lr = model_config["tabncd_classifier_lr"]
        tabncd_cluster_lr = model_config["tabncd_cluster_lr"]
        tabncd_k_neighbors = model_config["tabncd_k_neighbors"]
        tabncd_dropout = model_config["tabncd_dropout"]
        tabncd_activation_fct = model_config["tabncd_activation_fct"]

        clustering_prediction = None
        full_target_to_plot = None

        # ToDo implement the TabularNCD clustering model

        return jsonify({"error_message": "Model not implemented in the server"}), 422

    elif model_name == 'projection_in_classifier':

        model = ProjectionInClassifier(app,
                                       model_config['projection_in_classifier_architecture'],
                                       model_config['projection_in_classifier_n_clusters'],
                                       model_config['projection_in_classifier_dropout'],
                                       model_config['projection_in_classifier_activation_fct'],
                                       model_config['projection_in_classifier_lr'])

        batch_size = 256
        num_epochs = 30

        known_mask = np.in1d(np.array(dataset[target_name]), known_classes)
        x_train = filtered_dataset[known_mask]
        y_train = np.array(dataset[target_name])[known_mask]

        # The training targets must be in {0, ..., |C|}
        mapper, ind = np.unique(y_train, return_inverse=True)
        mapping_dict = dict(zip(y_train, ind))
        y_train_mapped = np.array(list(map(mapping_dict.get, y_train)))
        # y_test_known_mapped = np.array(list(map(mapping_dict.get, y_test_known)))

        # Start training in a new thread to avoid blocking the server while training the model
        global running_threads
        new_thread = NewThreadedTrainingTask(model, x_train, y_train_mapped, batch_size, num_epochs)
        new_thread.start()
        running_threads[new_thread.ident] = new_thread

        return jsonify({"thread_id": new_thread.ident}), 200
    else:
        return jsonify({"error_message": "Clustering method " + model_name + " not implemented yet"}), 422

    # Save the prediction of the clustering for future rules generation
    session['last_clustering_prediction'] = clustering_prediction.tolist()
    session['last_clustering_target_name'] = target_name
    session['last_clustering_original_target'] = np.array(dataset[target_name]).tolist()
    session['last_clustering_known_classes'] = known_classes
    session['last_clustering_unknown_classes'] = unknown_classes
    session['last_clustering_selected_features'] = selected_features

    image_folder_path = os.path.join('.', 'results', 'images_folder', dataset_name)

    if not os.path.isdir(image_folder_path):
        os.makedirs(image_folder_path)

    image_datetime_string = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")

    image_filename = image_datetime_string + '.png'

    fig = Figure(figsize=(8, 8))
    axis = fig.add_subplot(1, 1, 1)
    axis.set_title("T-SNE of the original " + dataset_name + " dataset colored by " + model_name)
    sns.scatterplot(ax=axis, x=np.array(tsne_array[tsne_array.columns[0]]), y=np.array(tsne_array[tsne_array.columns[1]]), hue=full_target_to_plot[unknown_mask])
    fig.savefig(os.path.join(image_folder_path, image_filename), dpi=fig.dpi, bbox_inches='tight')

    results_dict[dataset_name][corresponding_tsne_config_name]['images_configurations']['image_' + image_datetime_string] = {
        "image_configuration": {
            "random_state": random_state,
            "color_by": color_by,
            "model_config": model_config,
            "known_classes": known_classes,
            "unknown_classes": unknown_classes
        },
        "image_filepath": os.path.join(image_folder_path, image_filename),
    }

    saveResultsDict(results_dict)

    return send_file(os.path.join(image_folder_path, image_filename), mimetype='image/png')


@app.route('/runRulesGeneration', methods=['POST'])
def runRulesGeneration():
    if "loaded_dataset" not in session.keys():
        return jsonify({"error_message": "Dataset not loaded"}), 422

    dataset = session.get('loaded_dataset')

    last_clustering_prediction = session.get('last_clustering_prediction')
    last_clustering_target_name = session.get('last_clustering_target_name')
    last_clustering_original_target = session.get('last_clustering_original_target')
    last_clustering_known_classes = session.get('last_clustering_known_classes')
    last_clustering_unknown_classes = session.get('last_clustering_unknown_classes')
    last_clustering_selected_features = session.get('last_clustering_selected_features')

    if last_clustering_prediction is None:
        return jsonify({"error_message": "Please generate a prediction for the unknown classes"}), 422

    data = request.get_json()
    decision_tree_config = data['decision_tree_configuration']
    decision_tree_training_mode = decision_tree_config['decision_tree_training_mode']
    decision_tree_unknown_classes_only = decision_tree_config['decision_tree_unknown_classes_only']
    # decision_tree_max_depth = decision_tree_config['decision_tree_max_depth']
    # if decision_tree_max_depth == '':
    #     decision_tree_max_depth = None
    # elif type(decision_tree_max_depth) is str:
    #     decision_tree_max_depth = eval(decision_tree_max_depth)
    # decision_tree_min_samples_split = decision_tree_config['decision_tree_min_samples_split']
    # if decision_tree_min_samples_split == '':
    #     decision_tree_min_samples_split = None
    # elif type(decision_tree_min_samples_split) is str:
    #     decision_tree_min_samples_split = eval(decision_tree_min_samples_split)
    decision_tree_max_leaf_nodes = decision_tree_config['decision_tree_max_leaf_nodes']
    if decision_tree_max_leaf_nodes == '':
        decision_tree_max_leaf_nodes = None
    elif type(decision_tree_max_leaf_nodes) is str:
        decision_tree_max_leaf_nodes = eval(decision_tree_max_leaf_nodes)

    random_state = decision_tree_config['random_state']

    if decision_tree_unknown_classes_only is True:
        mask = np.in1d(np.array(dataset[last_clustering_target_name]), last_clustering_unknown_classes)

        x = dataset[last_clustering_selected_features][mask]
        y = last_clustering_prediction
    else:
        x = dataset[last_clustering_selected_features]
        y = np.array(last_clustering_original_target)
        y[np.in1d(np.array(dataset[last_clustering_target_name]), last_clustering_unknown_classes)] = last_clustering_prediction

    if decision_tree_training_mode == "multi_class":
        clf = DecisionTreeClassifier(max_leaf_nodes=decision_tree_max_leaf_nodes,
                                     random_state=random_state)
        clf.fit(x, y)
        accuracy_score = clf.score(x, y)

        dot_data = tree.export_graphviz(clf, out_file=None,
                                        feature_names=[f_n.replace('"', '') for f_n in last_clustering_selected_features],
                                        filled=True,
                                        max_depth=decision_tree_max_leaf_nodes,
                                        # class_names=le.inverse_transform(np.unique(label_enc)),
                                        proportion=False)

        dot_data = dot_data[:15] + 'label = "This tree has ' + "{:.1f}".format(accuracy_score*100) + '% train accuracy";\n' + dot_data[15:]

        graph = graphviz.Source(dot_data)
        filename = graph.render(os.path.join(".", "results", "latest_exported_tree"))  # export PDF

        return send_file(filename, mimetype='application/pdf')

    elif decision_tree_training_mode == "one_vs_rest":
        base_clf = DecisionTreeClassifier(max_leaf_nodes=decision_tree_max_leaf_nodes,
                                          random_state=random_state)
        clf = OneVsRestClassifier(base_clf)
        clf.fit(x, y)
        accuracy_score = clf.score(x, y)

        datetime_string = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")
        temp_folder = os.path.join(".", "results", "tmp", "temp_folder_" + datetime_string)

        pdfs_list = []
        for estimator, c in zip(clf.estimators_, clf.classes_):
            dot_data = tree.export_graphviz(estimator, out_file=None,
                                            feature_names=[f_n.replace('"', '') for f_n in last_clustering_selected_features],
                                            filled=True,
                                            max_depth=decision_tree_max_leaf_nodes,
                                            # class_names=['REST', le.inverse_transform(np.unique(label_enc))[elem]],
                                            proportion=False)

            class_or_cluster = "Class" if c in last_clustering_known_classes else "Cluster"
            dot_data = dot_data[:15] + 'label = "Tree for ' + class_or_cluster + ' ' + str(c) + ".\nWhole model had {:.1f}".format(accuracy_score*100) + '% average train accuracy";\n' + dot_data[15:]

            graph = graphviz.Source(dot_data)
            filename = graph.render(os.path.join(temp_folder, "class_" + str(c)))  # export PDF
            pdfs_list.append(filename)

        # Merge all the generated files into one
        merger = PdfMerger()
        for pdf_filename in pdfs_list:
            merger.append(pdf_filename)
        merged_filename = os.path.join(".", "results", "latest_exported_tree.pdf")
        merger.write(merged_filename)
        merger.close()

        # Remove the individual PDF files
        shutil.rmtree(temp_folder)

        return send_file(merged_filename, mimetype='application/pdf')
    else:
        return jsonify({"error_message": "Unknown decision tree training mode"}), 422


@app.route('/clearServerCache', methods=['POST'])
def clearServerCache():
    base_path = os.path.join('.', 'results')

    files_to_keep = [
        os.path.join(base_path, 'images_folder', '.gitkeep'),
        os.path.join(base_path, 'tsne_arrays', '.gitkeep')
    ]
    folders_to_keep = [
        os.path.join(base_path, 'images_folder'),
        os.path.join(base_path, 'tsne_arrays')
    ]

    files_to_delete = []
    folders_to_delete = []
    for root, dirs, files in os.walk(base_path):
        for name in files:
            full_file_path = os.path.join(root, name)
            if full_file_path not in files_to_keep:
                files_to_delete.append(full_file_path)
        for name in dirs:
            full_folder_path = os.path.join(root, name)
            if full_folder_path not in folders_to_keep:
                folders_to_delete.append(full_folder_path)

    # Delete all temporary files
    for file_path in files_to_delete:
        os.remove(file_path)

    # Remove now empty directories
    for folder_path in folders_to_delete:
        os.rmdir(folder_path)

    return "success", 200


@app.route('/getThreadProgress', methods=["POST"])
def getThreadProgress():
    data = request.get_json()
    if data["thread_id"] in running_threads.keys():
        return jsonify({"thread_progress": running_threads[data["thread_id"]].progress_percentage}), 200
    else:
        return jsonify({"error_message": "thread not running"}), 422


@app.route('/cancelTrainingThread', methods=["POST"])
def cancelTrainingThread():
    data = request.get_json()
    if data["thread_id"] in running_threads.keys():
        running_threads[data["thread_id"]].stop()
        del running_threads[data["thread_id"]]
        return "success", 200
    else:
        return jsonify({"error_message": "thread not running"}), 422

@app.errorhandler(500)
def internal_error(error):
    return jsonify({"error_message": str(error.original_exception)}), 422


if __name__ == '__main__':
    app.run(debug=True)
