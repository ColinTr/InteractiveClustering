/*
Software Name : InteractiveClustering
Version: 1.0
SPDX-FileCopyrightText: Copyright (c) 2024 Orange
SPDX-License-Identifier: MIT
This software is distributed under the MIT License,
the text of which is available at https://spdx.org/licenses/MIT.html
or see the "license.txt" file for more details.
*/

import React from 'react';
import Row from "react-bootstrap/Row";
import Col from "react-bootstrap/Col";
import Swal from "sweetalert2";
import {closeSnackbar, enqueueSnackbar} from 'notistack'
import ModelTrainingSnackbar from "./pop_up_notifiers/ModelTrainingSnackbar";
import ModelSelection from "./ModelSelection";
import DataVisualization from "./DataVisualization";
import DatasetSelector from "./DatasetSelector";
import FeatureSelection from "./FeatureSelection";
import RulesGenerator from "./RulesGenerator";
import FireSwalError from "./pop_up_notifiers/FireSwalError";
import FeatureDisplayModal from "./FeatureDisplayModal";
import WaitSnackbar from "./pop_up_notifiers/WaitSnackbar";


class FullPage extends React.Component {

    constructor(props) {
        super(props);

        // The states of the children that need to be shared are "lifted" here
        this.initial_state = {
            dataset_name : null,

            formatted_features : null,
            feature_search_query: '',
            search_filtered_features_list : null,

            selected_class_feature: null,
            class_values_to_display: null,
            unique_values_search_query: '',
            search_filtered_unique_values_list : null,

            image_to_display: null,
            show_unknown_only: false,

            n_features_used: null,
            n_known_classes: null,

            view_in_encoder_dict: {},
            estimated_times_to_train: {},

            // Default rules generation parameters
            decision_tree_training_mode: "one_vs_rest",
            decision_tree_unknown_classes_only: false,
            decision_tree_max_depth: null,
            decision_tree_min_samples_split: 2,
            decision_tree_max_leaf_nodes: 10,
            decision_tree_response_text_rules: "",
            decision_tree_response_pdf_file: null,
            decision_tree_response_accuracy_score: null,

            selected_model : "pbn",

            // Default PBN parameters
            model_pbn_epochs : 200,
            model_pbn_n_clusters : 10,
            model_pbn_w : 0.9,
            model_pbn_lr : 0.0005,
            model_pbn_dropout : 0.2,
            model_pbn_activation_fct : "relu",
            model_pbn_hidden_layers: [],

            // Default TabularNCD parameters
            model_tabncd_epochs : 50,
            model_tabncd_n_clusters : 10,
            model_tabncd_cosine_topk : 10,
            model_tabncd_w1 : 0.5,
            model_tabncd_w2 : 0.9,
            model_tabncd_topk : 0.5,
            model_tabncd_lr : 0.0005,
            model_tabncd_k_neighbors : 10,
            model_tabncd_dropout : 0.1,
            model_tabncd_activation_fct : "relu",
            model_tabncd_hidden_layers: [],

            // Default k means parameters
            model_k_means_n_clusters: 10,

            // Default spectral clustering parameters
            model_spectral_clustering_n_clusters: 10,
            model_spectral_clustering_affinity: 'rbf',

            // Default projection in classifier parameters
            model_projection_in_classifier_epochs: 200,
            model_projection_in_classifier_n_clusters: 10,
            model_projection_in_classifier_hidden_layers: [],
            model_projection_in_classifier_dropout: 0.2,
            model_projection_in_classifier_lr_value : 0.001,
            model_projection_in_classifier_activation_fct: "relu",
            model_projection_in_classifier_training_progress: 0,

            // Feature modal, opened when a point is clicked in the scatter plot
            feature_modal_is_open: false,
            point_feature_names: null,
            point_feature_values: null,
        };

        this.state = this.initial_state;
    }

    onFeatureRadioButtonChange = (feature_name) => {
        // Set the feature checkbox to true and disable it
        let formatted_features = [...this.state.formatted_features];  // Make a shallow copy
        this.state.search_filtered_features_list.forEach(feature => {  // For each feature currently displayed...
            formatted_features[feature.index].disabled = feature.name === feature_name;
            // if(feature.name === feature_name){
            //     formatted_features[feature.index].checked = true;
            // }
        })
        const updated_filtered_list = this.getUpdatedFilteredList(formatted_features, this.state.feature_search_query)
        this.setState({formatted_features: formatted_features, search_filtered_features_list: updated_filtered_list})  // And finally replace the array in the state
        this.setState({n_features_used: this.getNumberOfCheckedValues(formatted_features)})

        // Send request to the Flask server to display the unique values of this feature
        const requestOptions = {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                'dataset_name': this.state.dataset_name,
                'feature_name': feature_name})
        }
        fetch('/getFeatureUniqueValues', requestOptions)  // Don't need to specify the full localhost:5000/... as the proxy is set in package.json
            .then(serverPromise => {
                if (serverPromise.status === 500) {
                    FireSwalError('Status 500 - Server error', 'Please make sure that the server is running')
                }
                if (serverPromise.status === 422) {
                    serverPromise.json().then(error => {
                        FireSwalError('Status 422 - Server error', error['error_message'])
                    })
                }
                if (serverPromise.status === 200) {
                    serverPromise.json().then(response => {
                        this.setState({selected_class_feature: feature_name})
                        const new_formatted_class_values = response['unique_values'].map((feature, index) => ({"name": feature, "checked": true, "index": index, "used": true}))
                        this.setState({class_values_to_display: new_formatted_class_values})
                        this.setState({search_filtered_unique_values_list: this.getUpdatedFilteredList(new_formatted_class_values, this.state.unique_values_search_query)})
                        this.setState({n_known_classes: this.getNumberOfCheckedValues(new_formatted_class_values)})
                    })
                }
            })
    }

    onNewFeaturesLoaded = (new_features) => {
        const new_formatted_features = new_features.map((feature, index) => ({"name": feature, "checked": true, "index": index, "disabled": false}))
        this.setState({formatted_features: new_formatted_features})
        this.setState({search_filtered_features_list: this.getUpdatedFilteredList(new_formatted_features, this.state.feature_search_query)})
        const tmp_n_feat = this.getNumberOfCheckedValues(new_formatted_features)
        this.setState({n_features_used: tmp_n_feat})

        // Add default layers to models:
        this.setState({model_projection_in_classifier_hidden_layers: [Math.floor(0.75*tmp_n_feat), Math.floor(0.5*tmp_n_feat)]})
        this.setState({model_tabncd_hidden_layers: [Math.floor(0.75*tmp_n_feat), Math.floor(0.5*tmp_n_feat)]})
        this.setState({model_pbn_hidden_layers: [Math.floor(0.75*tmp_n_feat), Math.floor(0.5*tmp_n_feat)]})
    }

    onChangeFeaturesSearch = query => {
        this.setState({ feature_search_query: query.target.value })

        const updated_filtered_list = this.getUpdatedFilteredList(this.state.formatted_features, query.target.value)
        this.setState({ search_filtered_features_list: updated_filtered_list })
    };

    onChangeUniqueValuesSearch = query => {
        this.setState({ unique_values_search_query: query.target.value })

        const updated_filtered_list = this.getUpdatedFilteredList(this.state.class_values_to_display, query.target.value)
        this.setState({ search_filtered_unique_values_list: updated_filtered_list })
    };

    getUpdatedFilteredList = (features_list, query) => {
        if (features_list != null) {
            return features_list.filter((feature) => {
                if (query === "") {
                    return features_list
                } else {
                    return feature.name.toLowerCase().includes(query.toLowerCase())
                }
            })
        } else {
            return null
        }
    }

    getNumberOfCheckedValues = (list_of_values) => {
        if(list_of_values === null){
            return null
        }

        let count = 0
        list_of_values.map(feature => {
            // If the feature is 'disabled' it means that it is the class attribute, and we shouldn't count it
            if(Object.hasOwn(feature, 'disabled') === true){
                if(feature.disabled === false){
                    if (feature.checked === true) {
                        count += 1
                    }
                }
            }
            // If the feature is not 'used', it means that it is a class value that is unused, and we shouldn't count it
            if(Object.hasOwn(feature, 'used') === true){
                if (feature.used === true) {
                    if (feature.checked === true) {
                        count += 1
                    }
                }
            }
        })
        return count
    }

    onChangeCheckbox = i => {
        let formatted_features = [...this.state.formatted_features];  // Make a shallow copy
        let formatted_features_item = {...formatted_features[i]};  // Get the element we want to update
        formatted_features_item.checked = !formatted_features_item.checked  // Change it
        formatted_features[i] = formatted_features_item // Replace it in the array's copy

        const updated_filtered_list = this.getUpdatedFilteredList(formatted_features, this.state.feature_search_query)
        this.setState({formatted_features: formatted_features, search_filtered_features_list: updated_filtered_list})  // And finally replace the array in the state
        this.setState({n_features_used: this.getNumberOfCheckedValues(formatted_features)})

        if(formatted_features_item.name === this.state.selected_class_feature){
            this.setState({selected_class_feature: null})
            this.setState({class_values_to_display: null})
            this.setState({search_filtered_unique_values_list: null})

        }
    }

    onUniqueValueSwitchChange = i => {
        let class_values_to_display = [...this.state.class_values_to_display];  // Make a shallow copy
        let class_values_to_display_item = {...class_values_to_display[i]};  // Get the element we want to update
        class_values_to_display_item.checked = !class_values_to_display_item.checked  // Change it
        class_values_to_display[i] = class_values_to_display_item // Replace it in the array's copy

        const updated_filtered_list = this.getUpdatedFilteredList(class_values_to_display, this.state.unique_values_search_query)
        this.setState({class_values_to_display: class_values_to_display, search_filtered_unique_values_list: updated_filtered_list})  // And finally replace the array in the state
        this.setState({n_known_classes: this.getNumberOfCheckedValues(class_values_to_display)})
    }

    onChangeClassValueCheckbox = i => {
        let class_values_to_display = [...this.state.class_values_to_display];  // Make a shallow copy
        let class_values_to_display_item = {...class_values_to_display[i]};  // Get the element we want to update
        class_values_to_display_item.used = !class_values_to_display_item.used  // Change it
        class_values_to_display[i] = class_values_to_display_item // Replace it in the array's copy

        const updated_filtered_list = this.getUpdatedFilteredList(class_values_to_display, this.state.unique_values_search_query)
        this.setState({class_values_to_display: class_values_to_display, search_filtered_unique_values_list: updated_filtered_list})  // And finally replace the array in the state
        this.setState({n_known_classes: this.getNumberOfCheckedValues(class_values_to_display)})
    }

    onSwitchAllOnButtonClick = () => {
        if (this.state.class_values_to_display != null) {
            let class_values_to_display = [...this.state.class_values_to_display];  // Make a shallow copy
            this.state.search_filtered_unique_values_list.forEach(feature => {  // For each feature currently displayed...
                class_values_to_display[feature.index].checked = true
            })

            const updated_filtered_list = this.getUpdatedFilteredList(class_values_to_display, this.state.unique_values_search_query)
            this.setState({class_values_to_display: class_values_to_display, search_filtered_unique_values_list: updated_filtered_list})  // And finally replace the array in the state
            this.setState({n_known_classes: this.getNumberOfCheckedValues(class_values_to_display)})
        }
    }

    onSwitchAllOffButtonClick = () => {
        if (this.state.class_values_to_display != null) {
            let class_values_to_display = [...this.state.class_values_to_display];  // Make a shallow copy
            this.state.search_filtered_unique_values_list.forEach(feature => {  // For each feature currently displayed...
                class_values_to_display[feature.index].checked = false
            })
            const updated_filtered_list = this.getUpdatedFilteredList(class_values_to_display, this.state.unique_values_search_query)
            this.setState({class_values_to_display: class_values_to_display, search_filtered_unique_values_list: updated_filtered_list})  // And finally replace the array in the state
            this.setState({n_known_classes: this.getNumberOfCheckedValues(class_values_to_display)})
        }
    }

    onUncheckAllButtonClick = () => {
        if (this.state.formatted_features != null) {
            let formatted_features = [...this.state.formatted_features];  // Make a shallow copy
            this.state.search_filtered_features_list.forEach(feature => {  // For each feature currently displayed...
                formatted_features[feature.index].checked = false
            })
            const updated_filtered_list = this.getUpdatedFilteredList(formatted_features, this.state.feature_search_query)
            this.setState({formatted_features: formatted_features, search_filtered_features_list: updated_filtered_list})  // And finally replace the array in the state
            this.setState({n_features_used: this.getNumberOfCheckedValues(formatted_features)})
            this.setState({selected_class_feature: null})
            this.setState({class_values_to_display: null})
            this.setState({search_filtered_unique_values_list: null})
        }
    }

    onCheckAllButtonClick = () => {
        if (this.state.formatted_features != null) {
            let formatted_features = [...this.state.formatted_features];  // Make a shallow copy
            this.state.search_filtered_features_list.forEach(feature => {  // For each feature currently displayed...
                formatted_features[feature.index].checked = true
            })
            const updated_filtered_list = this.getUpdatedFilteredList(formatted_features, this.state.feature_search_query)
            this.setState({formatted_features: formatted_features, search_filtered_features_list: updated_filtered_list})  // And finally replace the array in the state
            this.setState({n_features_used: this.getNumberOfCheckedValues(formatted_features)})
        }
    }

    onClearFeaturesSearchButtonClick = () => {
        const updated_filtered_list = this.getUpdatedFilteredList(this.state.formatted_features, '')
        this.setState({search_filtered_features_list: updated_filtered_list, feature_search_query: ''})  // And finally replace the array in the state
    }

    onClearUniqueValuesSearchButtonClick = () => {
        const updated_filtered_list = this.getUpdatedFilteredList(this.state.class_values_to_display, '')
        this.setState({search_filtered_unique_values_list: updated_filtered_list, unique_values_search_query: ''})
    }

    getSelectedFeaturesFormattedList = () => {
        const selected_features = []
        this.state.formatted_features.forEach(feature => {
            if(feature.checked === true && feature.name !== this.state.selected_class_feature){
                selected_features.push(feature.name)
            }
        })
        return selected_features
    }

    getKnownClassesFormattedList = () => {
        const known_classes = []
        this.state.class_values_to_display.forEach(unique_value => {
            if(unique_value.checked === true && unique_value.used === true){
                known_classes.push(unique_value.name)
            }
        })
        return known_classes
    }

    getUnknownClassesFormattedList = () => {
        const unknown_classes = []
        this.state.class_values_to_display.forEach(unique_value => {
            if(unique_value.checked === false && unique_value.used === true){
                unknown_classes.push(unique_value.name)
            }
        })
        return unknown_classes
    }

    onRawDataButtonClick = () => {
        // Sanity checks:
        if(this.state.formatted_features == null){
            FireSwalError("Please load a dataset to visualize")
            return
        }
        if(this.state.selected_class_feature == null){
            FireSwalError("Please select a target feature")
            return
        }
        if(this.state.show_unknown_only === true && this.getUnknownClassesFormattedList().length === 0){
            FireSwalError("No unknown classes to plot", "Try unchecking \"Show unknown classes only\"")
            return
        }

        const requestOptions = {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                'dataset_name': this.state.dataset_name,
                'tsne_config': {
                    'selected_features': this.getSelectedFeaturesFormattedList(),
                    'known_classes': this.getKnownClassesFormattedList(),
                    'unknown_classes': this.getUnknownClassesFormattedList(),
                    'target_name': this.state.selected_class_feature,
                    'show_unknown_only': this.state.show_unknown_only,
                    'view_in_encoder': false,
                    'tsne_seed' : 0,
                    'tsne_perplexity': 30.0
                },
                'image_config': {
                    'random_state': 0,
                    'color_by': 'known_only',
                    "model_config": "",
                    'known_classes': this.getKnownClassesFormattedList(),
                    'unknown_classes': this.getUnknownClassesFormattedList()
                }
            })
        }

        // Open the waiting notification on screen
        const key = enqueueSnackbar({
            anchorOrigin: { vertical: 'top', horizontal: 'left', },
            persist: true,
            content: (key) => <WaitSnackbar id={key} message="Computing t-SNE..."/>,
        })

        fetch('/getDatasetTSNE', requestOptions)   // Don't need to specify the full localhost:5000/... as the proxy is set in package.json
            .then(serverPromise => {
                if (serverPromise.status === 500) {
                    FireSwalError('Status 500 - Server error', 'Please make sure that the server is running')
                }
                if (serverPromise.status === 422) {
                    serverPromise.json().then(error => {
                        FireSwalError('Status 422 - Server error', error['error_message'])
                    })
                }
                if (serverPromise.status === 200) {
                    serverPromise.json().then(plotly_data => {
                        this.setState({image_to_display: plotly_data})
                    })
                    // serverPromise.blob().then(image_response_blob => {
                    //     const imageObjectURL = URL.createObjectURL(image_response_blob);
                    //     this.setState({image_to_display: imageObjectURL})
                    // })
                }
            })
            .finally(() => {
                closeSnackbar(key)
            })
    }

    setDatasetNameHandler = (dataset_name) => {
        this.setState({dataset_name: dataset_name})
    }

    unloadDatasetHandler = () => {
        this.setState(this.initial_state)
    }

    onRulesRunButtonClick = () => {
        // Build the request
        const requestOptions = {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },

            body: JSON.stringify({
                'dataset_name': this.state.dataset_name,
                'decision_tree_configuration': {
                    'decision_tree_training_mode': this.state.decision_tree_training_mode,
                    'decision_tree_unknown_classes_only': this.state.decision_tree_unknown_classes_only,
                    'decision_tree_max_depth': this.state.decision_tree_max_depth,
                    'decision_tree_min_samples_split': this.state.decision_tree_min_samples_split,
                    'decision_tree_max_leaf_nodes': this.state.decision_tree_max_leaf_nodes,
                    'random_state': 0,
                }
            })
        }

        // Open the waiting notification on screen
        const key = enqueueSnackbar({
            anchorOrigin: { vertical: 'top', horizontal: 'left', },
            persist: true,
            content: (key) => <WaitSnackbar id={key} message="Generating rules..."/>,
        })

        fetch('/runRulesGeneration', requestOptions)   // Don't need to specify the full localhost:5000/... as the proxy is set in package.json
            .then(serverPromise => {
                if (serverPromise.status === 500) {
                    FireSwalError('Status 500 - Server error', 'Please make sure that the server is running')
                }
                if (serverPromise.status === 422) {
                    serverPromise.json().then(error => {
                        FireSwalError('Status 422 - Server error', error['error_message'])
                    })
                }
                if (serverPromise.status === 200) {
                    serverPromise.blob().then(response_blob => {
                        const file = URL.createObjectURL(response_blob)

                        this.setState({decision_tree_response_pdf_file: file})

                        Swal.mixin({
                            toast: true,
                            position: 'top-end',
                            showConfirmButton: false,
                            timer: 3000,
                            timerProgressBar: true,
                            didOpen: (toast) => {
                                toast.addEventListener('mouseenter', Swal.stopTimer)
                                toast.addEventListener('mouseleave', Swal.resumeTimer)
                            }
                        }).fire({
                            icon: 'success',
                            title: 'Rules generated'
                        })
                    })
                }
            })
            .finally(() => {
                closeSnackbar(key)
            })
    }

    onShowRulesButtonClick = () => {
        if(this.state.decision_tree_response_pdf_file === null){
            FireSwalError("No rules to show", "Please run a clustering before")
        } else {
            window.open(this.state.decision_tree_response_pdf_file)
        }
    }

    onDecisionTreeRadioButtonChange = (decision_tree_training_mode) => {
        this.setState({decision_tree_training_mode: decision_tree_training_mode})
    }

    on_decision_tree_max_depth_change = (event) => {
        this.setState({decision_tree_max_depth: parseInt(event.target.value)})
    }

    on_decision_tree_min_samples_split_change = (event) => {
        this.setState({decision_tree_min_samples_split: parseInt(event.target.value)})
    }

    on_decision_tree_decision_tree_max_leaf_nodes_change = (event) => {
        this.setState({decision_tree_max_leaf_nodes: parseInt(event.target.value)})
    }

    on_projection_in_classifier_epochs_change = (event) => {
        this.setState({model_projection_in_classifier_epochs: parseInt(event.target.value)})
    }

    on_projection_in_classifier_n_clusters_change = (event) => {
        this.setState({model_projection_in_classifier_n_clusters: parseInt(event.target.value)})
    }

    on_projection_in_classifier_dropout_change = (event) => {
        this.setState({model_projection_in_classifier_dropout: parseFloat(event.target.value)})
    }

    on_projection_in_classifier_activation_fct_change = (event) => {
        this.setState({model_projection_in_classifier_activation_fct: event.target.value})
    }

    on_projection_in_classifier_lr_change = (event) => {
        this.setState({model_projection_in_classifier_lr_value: parseFloat(event.target.value)})
    }

    onShowUnknownOnlySwitchChange = () => {
        this.setState({show_unknown_only: !this.state.show_unknown_only})
    }

    // onAgglomerativeClusteringRunButtonClick = (agglomerative_clustering_value) => {
    //     FireSwalError("Not implemented yet!")
    // }

    // onAgglomerativeClusteringUpdateRulesButtonClick = () => {
    //     FireSwalError("Not implemented yet!")
    // }

    updateTimeEstimationDict = (thread_id, new_progress_value) => {
        let estimated_times_to_train_copy = {...this.state.estimated_times_to_train}

        if(estimated_times_to_train_copy[thread_id] === undefined){
            estimated_times_to_train_copy[thread_id] = {'previous_values': [0], 'estimated_remaining_time': null}
        }

        estimated_times_to_train_copy[thread_id]['previous_values'].push(new_progress_value)

        if(estimated_times_to_train_copy[thread_id]['previous_values'].length < 2 || [...new Set(estimated_times_to_train_copy[thread_id]['previous_values'])] === [0]){
            estimated_times_to_train_copy[thread_id]['estimated_remaining_time'] = '...'
        } else {
            let average_progress_per_second = 0
            for (let i = 1; i < estimated_times_to_train_copy[thread_id]['previous_values'].length; i++) {
                average_progress_per_second = average_progress_per_second + (estimated_times_to_train_copy[thread_id]['previous_values'][i] - estimated_times_to_train_copy[thread_id]['previous_values'][i - 1])
            }
            average_progress_per_second = average_progress_per_second / estimated_times_to_train_copy[thread_id]['previous_values'].length

            const last_progress_value = estimated_times_to_train_copy[thread_id]['previous_values'][estimated_times_to_train_copy[thread_id]['previous_values'].length - 1]

            const estimated_remaining_time = (100 - last_progress_value) / average_progress_per_second

            if(estimated_remaining_time === Infinity){
                estimated_times_to_train_copy[thread_id]['estimated_remaining_time'] = '...'
            } else {
                estimated_times_to_train_copy[thread_id]['estimated_remaining_time'] = estimated_remaining_time
            }
        }

        return estimated_times_to_train_copy
    }

    onRunModelButtonClick = () => {
        // Sanity checks:
        if(this.state.formatted_features == null){
            FireSwalError("Please load a dataset to visualize")
            return
        }
        if(this.state.selected_class_feature == null){
            FireSwalError("Please select a target feature")
            return
        }
        if(this.state.show_unknown_only === true && this.getUnknownClassesFormattedList().length === 0){
            FireSwalError("No unknown classes to plot", "Try unchecking \"Show unknown classes only\"")
            return
        }
        if(this.getUnknownClassesFormattedList().length === 0){
            FireSwalError("Cannot run clustering", "There are no unknown classes selected")
            return
        }
        if((this.state.selected_model === "projection_in_classifier" || this.state.selected_model === "tabularncd") && this.getKnownClassesFormattedList().length === 0){
            FireSwalError("Cannot run clustering", "There are no known classes selected")
            return
        }
        if(this.state.selected_model === "projection_in_classifier"){
            if(this.state.model_projection_in_classifier_hidden_layers.length === 0){
                FireSwalError("Please add at least one hidden layer")
                return
            }
        }
        if(this.state.selected_model === "pbn"){
            if(this.state.model_pbn_hidden_layers.length === 0){
                FireSwalError("Please add at least one hidden layer")
                return
            }
        }

        let model_config = null

        if(this.state.selected_model === "pbn"){
            model_config = {
                'model_name': this.state.selected_model,

                'pbn_epochs': this.state.model_pbn_epochs,
                'pbn_n_clusters': parseInt(this.state.model_pbn_n_clusters),
                'pbn_w': parseFloat(this.state.model_pbn_w),
                'pbn_lr': parseFloat(this.state.model_pbn_lr),
                'pbn_dropout': parseFloat(this.state.model_pbn_dropout),
                'pbn_activation_fct': this.state.model_pbn_activation_fct,
                'input_size': this.state.n_features_used,
                'pbn_hidden_layers': this.state.model_pbn_hidden_layers
            }
        }
        else if(this.state.selected_model === "tabularncd"){
            model_config = {
                'model_name': this.state.selected_model,

                'tabncd_epochs': parseInt(this.state.model_tabncd_epochs),
                'tabncd_n_clusters': parseInt(this.state.model_tabncd_n_clusters),
                // 'tabncd_cosine_topk': parseFloat(this.state.model_tabncd_cosine_topk),
                'tabncd_w1': parseFloat(this.state.model_tabncd_w1),
                'tabncd_w2': parseFloat(this.state.model_tabncd_w2),
                'tabncd_lr': parseFloat(this.state.model_tabncd_lr),
                'tabncd_topk': parseFloat(this.state.model_tabncd_topk),
                'tabncd_k_neighbors': parseInt(this.state.model_tabncd_k_neighbors),
                'tabncd_dropout': parseFloat(this.state.model_tabncd_dropout),
                'tabncd_activation_fct': this.state.model_tabncd_activation_fct,
                'input_size': this.state.n_features_used,
                'tabncd_hidden_layers': this.state.model_tabncd_hidden_layers
            }
        }
        else if(this.state.selected_model === "k_means"){
            model_config = {
                'model_name': this.state.selected_model,

                'k_means_n_clusters': parseInt(this.state.model_k_means_n_clusters),
            }
        }
        else if(this.state.selected_model === "spectral_clustering") {
            model_config = {
                'model_name': this.state.selected_model,

                'spectral_clustering_n_clusters': parseInt(this.state.model_spectral_clustering_n_clusters),
                'spectral_clustering_affinity': this.state.model_spectral_clustering_affinity,
            }
        }
        else if(this.state.selected_model === "projection_in_classifier"){
            model_config = {
                'model_name': this.state.selected_model,

                'projection_in_classifier_epochs' : parseInt(this.state.model_projection_in_classifier_epochs),
                'projection_in_classifier_n_clusters' : parseInt(this.state.model_projection_in_classifier_n_clusters),
                'projection_in_classifier_architecture' : Array.prototype.concat(
                    this.state.n_features_used,
                    this.state.model_projection_in_classifier_hidden_layers,
                    this.state.n_known_classes),
                'projection_in_classifier_dropout' : parseFloat(this.state.model_projection_in_classifier_dropout),
                'projection_in_classifier_lr' : parseFloat(this.state.model_projection_in_classifier_lr_value),
                'projection_in_classifier_activation_fct' : this.state.model_projection_in_classifier_activation_fct,
            }
        }
        else {
            FireSwalError("Model not implemented yet")
            return
        }

        console.log(model_config)

        // Build the request
        const requestOptions = {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },

            body: JSON.stringify({
                'dataset_name': this.state.dataset_name,
                'tsne_config': {
                    'selected_features': this.getSelectedFeaturesFormattedList(),
                    'known_classes': this.getKnownClassesFormattedList(),
                    'unknown_classes': this.getUnknownClassesFormattedList(),
                    'target_name': this.state.selected_class_feature,
                    'show_unknown_only': this.state.show_unknown_only,
                    'view_in_encoder': false,
                    'tsne_seed' : 0,
                    'tsne_perplexity': 30.0
                },
                'image_config': {
                    'random_state': 0,
                    'color_by': 'model_prediction',
                    'model_config': model_config,
                    'known_classes': this.getKnownClassesFormattedList(),
                    'unknown_classes': this.getUnknownClassesFormattedList()
                }
            })
        }

        let clustering_model_key = ""
        if(this.state.selected_model === "k_means" || this.state.selected_model === "spectral_clustering"){
            // Open the waiting notification on screen
            clustering_model_key = enqueueSnackbar({
                anchorOrigin: { vertical: 'top', horizontal: 'left', },
                persist: true,
                content: (clustering_model_key) => <WaitSnackbar id={clustering_model_key} message="Fitting clustering model..."/>,
            })
        }

        fetch('/runClustering', requestOptions)   // Don't need to specify the full localhost:5000/... as the proxy is set in package.json
            .then(serverPromise => {
                if (serverPromise.status === 500) {
                    FireSwalError('Status 500 - Server error', 'Please make sure that the server is running')
                }
                if (serverPromise.status === 422) {
                    serverPromise.json().then(error => {
                        FireSwalError('Status 422 - Server error', error['error_message'])
                    })
                }
                if (serverPromise.status === 200) {
                    // These models take time to train, so we only get the background thread ID to update a progress bar
                    if(this.state.selected_model === "projection_in_classifier" || this.state.selected_model === "tabularncd" || this.state.selected_model === "pbn"){
                        serverPromise.json().then((server_response => {
                            const thread_id = server_response["thread_id"]

                            // Periodically request the backend during the training progress
                            const checkProgressRequestOptions = {
                                method: 'POST',
                                headers: { 'Content-Type': 'application/json' },
                                body: JSON.stringify({'thread_id': thread_id})
                            }
                            const refreshIntervalId = setInterval(() => fetch('/getThreadProgress', checkProgressRequestOptions)
                                .then(progressServerPromise => {
                                    if (progressServerPromise.status === 500) {
                                        FireSwalError('Status 500 - Server error', 'Please make sure that the server is running')
                                        clearInterval(refreshIntervalId)
                                    }
                                    if (progressServerPromise.status === 422) {
                                        progressServerPromise.json().then(error => {
                                            FireSwalError('Status 422 - Server error', error['error_message'])
                                            clearInterval(refreshIntervalId)
                                        })
                                    }
                                    if (progressServerPromise.status === 200) {
                                        progressServerPromise.json().then(json_response => {
                                            const progress_value = json_response["thread_progress"]
                                            this.setState({model_projection_in_classifier_training_progress: progress_value})

                                            const estimated_times_to_train_copy = this.updateTimeEstimationDict(thread_id, progress_value)
                                            this.setState({estimated_times_to_train: estimated_times_to_train_copy})

                                            const thread_estimated_remaining_time = estimated_times_to_train_copy[thread_id]['estimated_remaining_time']

                                            let formatted_remaining_time = ""
                                            if(thread_estimated_remaining_time === "..."){
                                                formatted_remaining_time = "..."
                                            } else {
                                                const hours = Math.floor(thread_estimated_remaining_time / (60 * 60))
                                                const minutes = Math.floor(thread_estimated_remaining_time / 60)
                                                const seconds = Math.round(Math.floor(thread_estimated_remaining_time % 60))

                                                if(hours !== 0){
                                                    if (hours.toString().length === 1) { formatted_remaining_time += "0" + hours + ":" }
                                                    else { formatted_remaining_time += hours + ":" }
                                                }
                                                if (minutes.toString().length === 1) { formatted_remaining_time += "0" + minutes + ":" }
                                                else { formatted_remaining_time += minutes + ":" }
                                                if (seconds.toString().length === 1) { formatted_remaining_time += "0" + seconds }
                                                else { formatted_remaining_time += seconds }
                                            }

                                            document.getElementById("time_estimation_thread_" + thread_id).innerHTML = "Remaining time: " + formatted_remaining_time
                                            document.getElementById("pb_thread_" + thread_id).setAttribute('aria-valuenow', progress_value)
                                            document.getElementById("pb_thread_" + thread_id).setAttribute('style','width:'+ Number(progress_value)+'%')

                                            // If the training is complete, we can stop asking for updates
                                            if(progress_value >= 100) {
                                                clearInterval(refreshIntervalId)
                                            }
                                        })
                                    }
                                }),1000)

                            let view_in_encoder_dict_copy = {...this.state.view_in_encoder_dict}
                            view_in_encoder_dict_copy[thread_id] = false
                            this.setState({view_in_encoder_dict: view_in_encoder_dict_copy})
                            // Open the download notification on screen
                            enqueueSnackbar({
                                anchorOrigin: { vertical: 'top', horizontal: 'left', },
                                persist: true,
                                content: (key, message) => <ModelTrainingSnackbar id={key}
                                                                                  message={message}
                                                                                  thread_id={thread_id}
                                                                                  refreshIntervalId={refreshIntervalId}
                                                                                  onSeeResultsButtonClick={this.onSeeResultsButtonClick}
                                                                                  onViewInEncoderSwitchChange={this.onViewInEncoderSwitchChange}/>,
                            })
                        }))
                    // Other clustering models are fast, so we just wait for the result
                    } else {
                        serverPromise.json()
                            .then(plotly_data => {
                                this.setState({image_to_display: plotly_data})
                            })
                            .finally(() => {
                                closeSnackbar(clustering_model_key)
                            })
                        // serverPromise.blob().then(image_response_blob => {
                        //     const imageObjectURL = URL.createObjectURL(image_response_blob);
                        //     this.setState({image_to_display: imageObjectURL})
                        // })

                    }
                }
            })
    }

    onViewInEncoderSwitchChange = (thread_id) => {
        let view_in_encoder_dict_copy = {...this.state.view_in_encoder_dict}
        view_in_encoder_dict_copy[thread_id] = !view_in_encoder_dict_copy[thread_id]
        this.setState({view_in_encoder_dict: view_in_encoder_dict_copy})
    }

    onAutoParamsButtonClick = () => {
        console.log("ToDo auto params for " + this.state.selected_model)
        FireSwalError("Not implemented yet!")
    }

    updateSelectedModel = (model_name) => {
        this.setState({ selected_model: model_name })
    }

    on_kmeans_n_clusters_change = (event) => {
        this.setState({model_k_means_n_clusters: parseInt(event.target.value)})
    }

    on_pbn_epochs_change = (event) => {
        this.setState({model_pbn_epochs: parseInt(event.target.value)})
    }

    on_pbn_n_clusters_change = (event) => {
        this.setState({model_pbn_n_clusters: parseInt(event.target.value)})
    }

    on_pbn_w_change = (event) => {
        this.setState({model_pbn_w: parseFloat(event.target.value)})
    }

    on_pbn_lr_change = (event) => {
        this.setState({model_pbn_lr: parseFloat(event.target.value)})
    }

    on_pbn_dropout_change = (event) => {
        this.setState({model_pbn_dropout: parseFloat(event.target.value)})
    }

    on_pbn_activation_fct_change = (event) => {
        this.setState({model_pbn_activation_fct: event.target.value})
    }

    on_tabncd_epochs_change = (event) => {
        this.setState({model_tabncd_epochs: parseInt(event.target.value)})
    }

    on_tabncd_n_clusters_change = (event) => {
        this.setState({model_tabncd_n_clusters: parseInt(event.target.value)})
    }

    on_tabncd_cosine_topk_change = (event) => {
        this.setState({model_tabncd_cosine_topk: parseFloat(event.target.value)})
    }

    on_tabncd_w1_change = (event) => {
        this.setState({model_tabncd_w1: parseFloat(event.target.value)})
    }

    on_tabncd_w2_change = (event) => {
        this.setState({model_tabncd_w2: parseFloat(event.target.value)})
    }

    on_tabncd_lr_change = (event) => {
        this.setState({model_tabncd_lr: parseFloat(event.target.value)})
    }

    on_tabncd_topk_change = (event) => {
        this.setState({model_tabncd_topk: parseFloat(event.target.value)})
    }

    on_tabncd_k_neighbors_change = (event) => {
        this.setState({model_tabncd_k_neighbors: parseInt(event.target.value)})
    }

    on_tabncd_dropout_change = (event) => {
        this.setState({model_tabncd_dropout: parseFloat(event.target.value)})
    }

    on_tabncd_activation_fct_change = (event) => {
        this.setState({model_tabncd_activation_fct: event.target.value})
    }

    on_spectral_clustering_n_clusters_change = (event) => {
        this.setState({model_spectral_clustering_n_clusters: parseInt(event.target.value)})
    }

    on_spectral_clustering_affinity_change = (event) => {
        this.setState({model_spectral_clustering_affinity: event.target.value})
    }

    onRulesUnknownClassesOnlySwitchChange = () => {
        this.setState({decision_tree_unknown_classes_only: !this.state.decision_tree_unknown_classes_only})
    }

    closeFeaturesModal = () => {
        this.setState({feature_modal_is_open: false})
    }

    onSaveImageButtonClick = () => {
        if(this.state.image_to_display !== null) {
            const link = document.createElement('a')
            link.href = this.state.image_to_display
            link.setAttribute('download', this.state.dataset_name + '.png')
            document.body.appendChild(link)
            link.click()
            link.parentNode.removeChild(link)  // Clean up and remove the link
        } else {
            FireSwalError("No image to save.")
        }
    }

    onProjectionInClassifierAddLayerButtonClick = () => {
        const layer_size = document.getElementById('projectionInClassifierLayerSizeInput').value

        if(layer_size === null || layer_size === '' || layer_size <= 0){
            FireSwalError("Please enter a valid value")
            return
        }

        let model_projection_in_classifier_hidden_layers_copy = [...this.state.model_projection_in_classifier_hidden_layers]
        model_projection_in_classifier_hidden_layers_copy.push(parseInt(layer_size))
        this.setState({model_projection_in_classifier_hidden_layers: model_projection_in_classifier_hidden_layers_copy})
    }

    onProjectionInClassifierRemoveLayerButtonClick = (layer_index) => {
        let model_projection_in_classifier_hidden_layers_copy = [...this.state.model_projection_in_classifier_hidden_layers]
        model_projection_in_classifier_hidden_layers_copy.splice(layer_index, 1)
        this.setState({model_projection_in_classifier_hidden_layers: model_projection_in_classifier_hidden_layers_copy})
    }

    onTabncdAddLayerButtonClick = () => {
        const layer_size = document.getElementById('tabncdLayerSizeInput').value

        if(layer_size === null || layer_size === '' || layer_size <= 0){
            FireSwalError("Please enter a valid value")
            return
        }

        let model_tabncd_hidden_layers_copy = [...this.state.model_tabncd_hidden_layers]
        model_tabncd_hidden_layers_copy.push(parseInt(layer_size))
        this.setState({model_tabncd_hidden_layers: model_tabncd_hidden_layers_copy})
    }

    onTabncdRemoveLayerButtonClick = (layer_index) => {
        let model_tabncd_hidden_layers_copy = [...this.state.model_tabncd_hidden_layers]
        model_tabncd_hidden_layers_copy.splice(layer_index, 1)
        this.setState({model_tabncd_hidden_layers: model_tabncd_hidden_layers_copy})
    }

    on_pbn_add_layer_button_click = () => {
        const layer_size = document.getElementById('pbnLayerSizeInput').value

        if(layer_size === null || layer_size === '' || layer_size <= 0){
            FireSwalError("Please enter a valid value")
            return
        }

        let model_pbn_hidden_layers_copy = [...this.state.model_pbn_hidden_layers]
        model_pbn_hidden_layers_copy.push(parseInt(layer_size))
        this.setState({model_pbn_hidden_layers: model_pbn_hidden_layers_copy})
    }

    on_pbn_remove_layer_button_click = (layer_index) => {
        let model_pbn_hidden_layers_copy = [...this.state.model_pbn_hidden_layers]
        model_pbn_hidden_layers_copy.splice(layer_index, 1)
        this.setState({model_pbn_hidden_layers: model_pbn_hidden_layers_copy})
    }

    onSeeResultsButtonClick = (thread_id) => {
        const requestOptions = {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },

            body: JSON.stringify({
                'thread_id': thread_id,
                'show_unknown_only': this.state.show_unknown_only,
                'view_in_encoder': this.state.view_in_encoder_dict[thread_id]
            })
        }

        // Open the waiting notification on screen
        const key = enqueueSnackbar({
            anchorOrigin: { vertical: 'top', horizontal: 'left', },
            persist: true,
            content: (key) => <WaitSnackbar id={key} message="Computing t-SNE..."/>,
        })

        fetch('/getThreadResults', requestOptions)   // Don't need to specify the full localhost:5000/... as the proxy is set in package.json
            .then(serverPromise => {
                if (serverPromise.status === 500) {
                    FireSwalError('Status 500 - Server error', 'Please make sure that the server is running')
                }
                if (serverPromise.status === 422) {
                    serverPromise.json().then(error => {
                        FireSwalError('Status 422 - Server error', error['error_message'])
                    })
                }
                if (serverPromise.status === 200) {
                    serverPromise.json().then(plotly_data => {
                        this.setState({image_to_display: plotly_data})
                    })
                    // serverPromise.blob().then(image_response_blob => {
                    //     const imageObjectURL = URL.createObjectURL(image_response_blob);
                    //     this.setState({image_to_display: imageObjectURL})
                    // })
                }
            })
            .finally(() => {
                closeSnackbar(key)
            })
    }

    onClearCacheButtonClick = () => {
        Swal.fire({
            title: 'Are you sure?',
            text: "Clearing the server\'s temporary files might increase the processing time of the next requests.",
            showDenyButton: true,
            confirmButtonText: 'Clear',
            denyButtonText: `Don't clear`,
        }).then((result) => {
            if (result.isConfirmed) {
                const requestOptions = {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' }
                }
                fetch('/clearServerCache', requestOptions)
                    .then(serverPromise => {
                        if (serverPromise.status === 500) {
                            FireSwalError('Status 500 - Server error', 'Please make sure that the server is running')
                        }
                        if (serverPromise.status === 422) {
                            serverPromise.json().then(error => {
                                FireSwalError('Status 422 - Server error', error['error_message'])
                            })
                        }
                        if (serverPromise.status === 200) {
                            Swal.fire('Done!', '', 'success')

                            this.setState(this.initial_state)

                            // closeSnackbar()  // Closes all opened snackbars
                        }
                    })
            }
        })
    }

    handlePointClick = e => {
        const point_to_get = e['points'][0]

        const requestOptions = {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },

            body: JSON.stringify({
                'dataset_name': this.state.dataset_name,
                'class_name': point_to_get.data.name,
                'point_number': point_to_get.pointNumber
            })
        }
        fetch('/getPointData', requestOptions)
            .then(serverPromise => {
                if (serverPromise.status === 500) {
                    FireSwalError('Status 500 - Server error', 'Please make sure that the server is running')
                }
                if (serverPromise.status === 422) {
                    serverPromise.json().then(error => {
                        FireSwalError('Status 422 - Server error', error['error_message'])
                    })
                }
                if (serverPromise.status === 200) {
                    serverPromise.json().then(server_json_response => {
                        console.log(server_json_response)
                        this.setState({
                            point_feature_names: server_json_response.point_feature_names,
                            point_feature_values: server_json_response.point_feature_values,
                            feature_modal_is_open: true
                        })
                    })
                }
            })
    }

    render() {
        return (
            <Row style={{height: '100%', width:"99%"}} className="d-flex flex-row justify-content-center align-items-center">
                <FeatureDisplayModal feature_modal_is_open={this.state.feature_modal_is_open}
                                     closeFeaturesModal={this.closeFeaturesModal}
                                     point_feature_names={this.state.point_feature_names}
                                     point_feature_values={this.state.point_feature_values}
                />

                <Col className="col-lg-3 col-12 d-flex flex-column" style={{height: "98%"}}>
                    <Row className="my_row py-2">
                        <DatasetSelector onNewFeaturesLoaded={this.onNewFeaturesLoaded}
                                         setDatasetNameHandler={this.setDatasetNameHandler}
                                         unloadDatasetHandler={this.unloadDatasetHandler}
                        />
                    </Row>
                    <Row className="my_row mb-lg-0 py-2" style={{flexGrow:'1'}}>
                        <FeatureSelection feature_search_query={this.state.feature_search_query}
                                          onChangeFeaturesSearch={this.onChangeFeaturesSearch}
                                          search_filtered_features_list={this.state.search_filtered_features_list}
                                          selected_class_feature={this.state.selected_class_feature}

                                          onClearFeaturesSearchButtonClick={this.onClearFeaturesSearchButtonClick}
                                          onCheckAllButtonClick={this.onCheckAllButtonClick}
                                          onUncheckAllButtonClick={this.onUncheckAllButtonClick}
                                          onChangeCheckbox={this.onChangeCheckbox}
                                          onFeatureRadioButtonChange={this.onFeatureRadioButtonChange}

                                          unique_values_search_query={this.state.unique_values_search_query}
                                          onChangeUniqueValuesSearch={this.onChangeUniqueValuesSearch}
                                          search_filtered_unique_values_list={this.state.search_filtered_unique_values_list}

                                          onClearUniqueValuesSearchButtonClick={this.onClearUniqueValuesSearchButtonClick}
                                          onSwitchAllOnButtonClick={this.onSwitchAllOnButtonClick}
                                          onSwitchAllOffButtonClick={this.onSwitchAllOffButtonClick}
                                          onUniqueValueSwitchChange={this.onUniqueValueSwitchChange}
                                          onChangeClassValueCheckbox={this.onChangeClassValueCheckbox}
                        />
                    </Row>
                </Col>

                <Col className="col-lg-6 col-12 d-flex flex-column justify-content-center" style={{height: "98%"}}>
                    <Row className="my_row mx-lg-0 mb-lg-0 py-2 d-flex flex-row" style={{flexGrow:'1', height:"100%"}}>
                        <DataVisualization image_to_display={this.state.image_to_display}
                                           onRawDataButtonClick={this.onRawDataButtonClick}

                                           onShowUnknownOnlySwitchChange={this.onShowUnknownOnlySwitchChange}
                                           show_unknown_only={this.state.show_unknown_only}

                                           onSaveImageButtonClick={this.onSaveImageButtonClick}
                                           onClearCacheButtonClick={this.onClearCacheButtonClick}

                                           handlePointClick={this.handlePointClick}
                        />
                    </Row>
                </Col>

                <Col className="col-lg-3 col-12 d-flex flex-column" style={{height: "98%"}}>
                    <Row className="my_row py-2 d-flex flex-row" style={{flexGrow:'1'}}>
                        <ModelSelection onRunModelButtonClick={this.onRunModelButtonClick}
                                        onAutoParamsButtonClick={this.onAutoParamsButtonClick}
                                        updateSelectedModel={this.updateSelectedModel}

                                        n_features_used={this.state.n_features_used}
                                        n_known_classes={this.state.n_known_classes}

                                        on_pbn_epochs_change={this.on_pbn_epochs_change}
                                        pbn_epochs={this.state.model_pbn_epochs}
                                        on_pbn_n_clusters_change={this.on_pbn_n_clusters_change}
                                        pbn_n_clusters={this.state.model_pbn_n_clusters}
                                        on_pbn_w_change={this.on_pbn_w_change}
                                        pbn_w={this.state.model_pbn_w}
                                        on_pbn_lr_change={this.on_pbn_lr_change}
                                        pbn_lr={this.state.model_pbn_lr}
                                        on_pbn_dropout_change={this.on_pbn_dropout_change}
                                        pbn_dropout={this.state.model_pbn_dropout}
                                        on_pbn_activation_fct_change={this.on_pbn_activation_fct_change}
                                        pbn_hidden_layers={this.state.model_pbn_hidden_layers}
                                        on_pbn_add_layer_button_click={this.on_pbn_add_layer_button_click}
                                        on_pbn_remove_layer_button_click={this.on_pbn_remove_layer_button_click}

                                        on_tabncd_epochs_change={this.on_tabncd_epochs_change}
                                        tabncd_epochs={this.state.model_tabncd_epochs}
                                        on_tabncd_n_clusters_change={this.on_tabncd_n_clusters_change}
                                        tabncd_n_clusters={this.state.model_tabncd_n_clusters}
                                        on_tabncd_cosine_topk_change={this.on_tabncd_cosine_topk_change}
                                        tabncd_cosine_topk={this.state.model_tabncd_cosine_topk}
                                        on_tabncd_w1_change={this.on_tabncd_w1_change}
                                        tabncd_w1={this.state.model_tabncd_w1}
                                        on_tabncd_w2_change={this.on_tabncd_w2_change}
                                        tabncd_w2={this.state.model_tabncd_w2}
                                        on_tabncd_topk_change={this.on_tabncd_topk_change}
                                        tabncd_topk={this.state.model_tabncd_topk}
                                        on_tabncd_lr_change={this.on_tabncd_lr_change}
                                        tabncd_lr={this.state.model_tabncd_lr}
                                        on_tabncd_k_neighbors_change={this.on_tabncd_k_neighbors_change}
                                        tabncd_k_neighbors={this.state.model_tabncd_k_neighbors}
                                        on_tabncd_dropout_change={this.on_tabncd_dropout_change}
                                        tabncd_dropout={this.state.model_tabncd_dropout}
                                        on_tabncd_activation_fct_change={this.on_tabncd_activation_fct_change}
                                        tabncd_hidden_layers={this.state.model_tabncd_hidden_layers}
                                        onTabncdAddLayerButtonClick={this.onTabncdAddLayerButtonClick}
                                        onTabncdRemoveLayerButtonClick={this.onTabncdRemoveLayerButtonClick}

                                        on_kmeans_n_clusters_change={this.on_kmeans_n_clusters_change}
                                        k_means_n_clusters={this.state.model_k_means_n_clusters}
                                        onKMeansTrainOnUnknownClassesOnlySwitchChange={this.onKMeansTrainOnUnknownClassesOnlySwitchChange}

                                        on_spectral_clustering_n_clusters_change={this.on_spectral_clustering_n_clusters_change}
                                        spectral_clustering_n_clusters={this.state.model_spectral_clustering_n_clusters}
                                        on_spectral_clustering_affinity_change={this.on_spectral_clustering_affinity_change}
                                        spectral_clustering_affinity={this.state.model_spectral_clustering_affinity}

                                        on_projection_in_classifier_epochs_change = {this.on_projection_in_classifier_epochs_change}
                                        projection_in_classifier_epochs = {this.state.model_projection_in_classifier_epochs}
                                        on_projection_in_classifier_n_clusters_change = {this.on_projection_in_classifier_n_clusters_change}
                                        projection_in_classifier_n_clusters = {this.state.model_projection_in_classifier_n_clusters}
                                        projection_in_classifier_hidden_layers = {this.state.model_projection_in_classifier_hidden_layers}
                                        on_projection_in_classifier_dropout_change = {this.on_projection_in_classifier_dropout_change}
                                        projection_in_classifier_dropout = {this.state.model_projection_in_classifier_dropout}
                                        on_projection_in_classifier_activation_fct_change = {this.on_projection_in_classifier_activation_fct_change}
                                        on_projection_in_classifier_lr_change = {this.on_projection_in_classifier_lr_change}
                                        projection_in_classifier_lr_value = {this.state.model_projection_in_classifier_lr_value}
                                        onProjectionInClassifierAddLayerButtonClick = {this.onProjectionInClassifierAddLayerButtonClick}
                                        onProjectionInClassifierRemoveLayerButtonClick = {this.onProjectionInClassifierRemoveLayerButtonClick}
                        />
                    </Row>
                    <Row className="my_row mb-lg-0 py-2 d-flex flex-row">
                        <RulesGenerator onDecisionTreeRadioButtonChange={this.onDecisionTreeRadioButtonChange}
                                        decision_tree_training_mode={this.state.decision_tree_training_mode}

                                        onRulesUnknownClassesOnlySwitchChange={this.onRulesUnknownClassesOnlySwitchChange}

                                        on_decision_tree_max_depth_change={this.on_decision_tree_max_depth_change}
                                        decision_tree_max_depth={this.state.decision_tree_max_depth}

                                        on_decision_tree_min_samples_split_change={this.on_decision_tree_min_samples_split_change}
                                        decision_tree_min_samples_split={this.state.decision_tree_min_samples_split}

                                        on_decision_tree_decision_tree_max_leaf_nodes_change={this.on_decision_tree_decision_tree_max_leaf_nodes_change}
                                        decision_tree_decision_tree_max_leaf_nodes={this.state.decision_tree_max_leaf_nodes}

                                        onRulesRunButtonClick={this.onRulesRunButtonClick}
                                        onShowRulesButtonClick={this.onShowRulesButtonClick}
                        />
                    </Row>
                    {/*
                    <Row className="my_row py-2 d-flex flex-row">
                        <AgglomerativeClustering onAgglomerativeClusteringRunButtonClick={this.onAgglomerativeClusteringRunButtonClick}
                                                 onAgglomerativeClusteringUpdateRulesButtonClick={this.onAgglomerativeClusteringUpdateRulesButtonClick}/>
                    </Row>
                    */}
                </Col>

            </Row>
        )
    }
}

export default FullPage;