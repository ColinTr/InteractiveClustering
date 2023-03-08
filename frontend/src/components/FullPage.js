import React from 'react';
import Row from "react-bootstrap/Row";
import Col from "react-bootstrap/Col";
import ModelSelection from "./ModelSelection";
import AgglomerativeClustering from "./AgglomerativeClustering";
import DataVisualization from "./DataVisualization";
import DatasetSelector from "./DatasetSelector";
import FeatureSelection from "./FeatureSelection";
import fireSwalError from "./swal_functions";
import RulesGenerator from "./RulesGenerator";
import RulesDisplayModal from "./RulesDisplayModal";
import Swal from "sweetalert2";


class FullPage extends React.Component {

    default_model_params = {
        // Default k means parameters :
        default_kmeans_n_clusters : 10,

        // Default spectral clustering parameters :
        default_spectral_clustering_n_clusters : 10,
        default_spectral_clustering_affinity : 'rbf',

        // Default TabularNCD parameters :
        default_tabncd_n_clusters : 10,
        default_tabncd_cosine_topk : 10,
        default_tabncd_w1 : 0.8,
        default_tabncd_w2 : 0.8,
        default_tabncd_classifier_lr : 0.001,
        default_tabncd_cluster_lr : 0.001,
        default_tabncd_k_neighbors : 5,
        default_tabncd_dropout : 0.2,
        default_tabncd_activation_fct : "Sigmoid"
    }

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

            // Rules generation parameters
            decision_tree_training_mode: "multi_class",
            decision_tree_unknown_classes_only: false,
            decision_tree_max_depth: null,
            decision_tree_min_samples_split: 2,
            decision_tree_max_leaf_nodes: 10,
            rules_modal_is_open: false,
            decision_tree_response_text_rules: "",
            decision_tree_response_pdf_file: null,
            decision_tree_response_accuracy_score: null,

            model_params_selected_model : "tabularncd",

            // k means parameters :
            model_params_k_means_n_clusters: this.default_model_params.default_kmeans_n_clusters,

            // spectral clustering parameters :
            model_params_spectral_clustering_n_clusters: this.default_model_params.default_spectral_clustering_n_clusters,
            model_params_spectral_clustering_affinity: this.default_model_params.default_spectral_clustering_affinity,

            // TabularNCD parameters :
            model_params_tabncd_n_clusters : this.default_model_params.default_tabncd_n_clusters,
            model_params_tabncd_cosine_topk : this.default_model_params.default_tabncd_cosine_topk,
            model_params_tabncd_w1 : this.default_model_params.default_tabncd_w1,
            model_params_tabncd_w2 : this.default_model_params.default_tabncd_w2,
            model_params_tabncd_classifier_lr : this.default_model_params.default_tabncd_classifier_lr,
            model_params_tabncd_cluster_lr : this.default_model_params.default_tabncd_cluster_lr,
            model_params_tabncd_k_neighbors : this.default_model_params.default_tabncd_k_neighbors,
            model_params_tabncd_dropout : this.default_model_params.default_tabncd_dropout,
            model_params_tabncd_activation_fct : this.default_model_params.default_tabncd_activation_fct,
        };

        this.state = this.initial_state;
    }

    onNewFeaturesLoaded = (new_features) => {
        const new_formatted_features = new_features.map((feature, index) => ({"name": feature, "checked" : true, index : index, "disabled": false}))
        this.setState({formatted_features: new_formatted_features})
        this.setState({search_filtered_features_list: this.getUpdatedFilteredList(new_formatted_features, this.state.feature_search_query)})
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

    onChangeCheckbox = i => {
        let formatted_features = [...this.state.formatted_features];  // Make a shallow copy
        let formatted_features_item = {...formatted_features[i]};  // Get the element we want to update
        formatted_features_item.checked = !formatted_features_item.checked  // Change it
        formatted_features[i] = formatted_features_item // Replace it in the array's copy

        const updated_filtered_list = this.getUpdatedFilteredList(formatted_features, this.state.feature_search_query)
        this.setState({formatted_features: formatted_features, search_filtered_features_list: updated_filtered_list})  // And finally replace the array in the state
    }

    onUniqueValueSwitchChange = i => {
        let class_values_to_display = [...this.state.class_values_to_display];  // Make a shallow copy
        let class_values_to_display_item = {...class_values_to_display[i]};  // Get the element we want to update
        class_values_to_display_item.checked = !class_values_to_display_item.checked  // Change it
        class_values_to_display[i] = class_values_to_display_item // Replace it in the array's copy

        const updated_filtered_list = this.getUpdatedFilteredList(class_values_to_display, this.state.unique_values_search_query)
        this.setState({class_values_to_display: class_values_to_display, search_filtered_unique_values_list: updated_filtered_list})  // And finally replace the array in the state
    }

    onSwitchAllOnButtonClick = () => {
        if (this.state.class_values_to_display != null) {
            let class_values_to_display = [...this.state.class_values_to_display];  // Make a shallow copy
            this.state.search_filtered_unique_values_list.forEach(feature => {  // For each feature currently displayed...
                class_values_to_display[feature.index].checked = true
            })

            const updated_filtered_list = this.getUpdatedFilteredList(class_values_to_display, this.state.unique_values_search_query)
            this.setState({class_values_to_display: class_values_to_display, search_filtered_unique_values_list: updated_filtered_list})  // And finally replace the array in the state
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
            if(unique_value.checked === true){
                known_classes.push(unique_value.name)
            }
        })
        return known_classes
    }

    getUnknownClassesFormattedList = () => {
        const unknown_classes = []
        this.state.class_values_to_display.forEach(unique_value => {
            if(unique_value.checked === false){
                unknown_classes.push(unique_value.name)
            }
        })
        return unknown_classes
    }

    onRawDataButtonClick = () => {
        // Sanity checks:
        if(this.state.formatted_features == null){
            fireSwalError("Please load a dataset to visualize")
            return
        }
        if(this.state.selected_class_feature == null){
            fireSwalError("Please select a target feature")
            return
        }
        if(this.state.show_unknown_only === true && this.getUnknownClassesFormattedList().length === 0){
            fireSwalError("No unknown classes to plot", "Try unchecking \"Show unknown classes only\"")
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
        fetch('/getDatasetTSNE', requestOptions)   // Don't need to specify the full localhost:5000/... as the proxy is set in package.json
            .then(serverPromise => {
                if (serverPromise.status === 500) {
                    fireSwalError('Status 500 - Server error', 'Please make sure that the server is running')
                }
                if (serverPromise.status === 422) {
                    serverPromise.json().then(error => {
                        fireSwalError('Status 422 - Server error', error['error_message'])
                    })
                }
                if (serverPromise.status === 200) {
                    serverPromise.blob().then(image_response_blob => {
                        const imageObjectURL = URL.createObjectURL(image_response_blob);
                        this.setState({image_to_display: imageObjectURL})
                    })
                }
            })
    }

    onProjectionButtonClick = () => {
        if(this.state.formatted_features == null){
            fireSwalError("Please load a dataset to visualize")
        } else {
            if(this.state.selected_class_feature == null){
                fireSwalError("Please select a target feature")
            } else {
                console.log("ToDo : launch T-SNE of raw data with Flask server...")
                fireSwalError("Not implemented yet!")
            }
        }
    }

    onFeatureRadioButtonChange = (feature_name) => {
        // Set the feature checkbox to true and disable it
        let formatted_features = [...this.state.formatted_features];  // Make a shallow copy
        this.state.search_filtered_features_list.forEach(feature => {  // For each feature currently displayed...
            formatted_features[feature.index].disabled = feature.name === feature_name;
        })
        const updated_filtered_list = this.getUpdatedFilteredList(formatted_features, this.state.feature_search_query)
        this.setState({formatted_features: formatted_features, search_filtered_features_list: updated_filtered_list})  // And finally replace the array in the state

        // Send request to the Flask server to display the unique values of this feature
        const requestOptions = {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({'feature_name': feature_name})
        }
        fetch('/getFeatureUniqueValues', requestOptions)   // Don't need to specify the full localhost:5000/... as the proxy is set in package.json
            .then(serverPromise => {
                if (serverPromise.status === 500) {
                    fireSwalError('Status 500 - Server error', 'Please make sure that the server is running')
                }
                if (serverPromise.status === 422) {
                    serverPromise.json().then(error => {
                        fireSwalError('Status 422 - Server error', error['error_message'])
                    })
                }
                if (serverPromise.status === 200) {
                    serverPromise.json().then(response => {
                        this.setState({selected_class_feature: feature_name})
                        const new_formatted_class_values = response['unique_values'].map((feature, index) => ({"name": feature, "checked" : true, index : index}))
                        this.setState({class_values_to_display: new_formatted_class_values})
                        this.setState({search_filtered_unique_values_list: this.getUpdatedFilteredList(new_formatted_class_values, this.state.unique_values_search_query)})
                    })
                }
            })
    }

    setDatasetNameHandler = (dataset_name) => {
        this.setState({dataset_name: dataset_name})
    }

    unloadDatasetHandler = () => {
        this.setState(this.initial_state)
    }

    onRulesRunButtonClick = () => {
        // ToDo sanity checks

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
        fetch('/runRulesGeneration', requestOptions)   // Don't need to specify the full localhost:5000/... as the proxy is set in package.json
            .then(serverPromise => {
                if (serverPromise.status === 500) {
                    fireSwalError('Status 500 - Server error', 'Please make sure that the server is running')
                }
                if (serverPromise.status === 422) {
                    serverPromise.json().then(error => {
                        fireSwalError('Status 422 - Server error', error['error_message'])
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

                    // serverPromise.json().then(response_json => {
                    //     this.setState({
                    //         decision_tree_response_training_mode: response_json["decision_tree_training_mode"],
                    //         decision_tree_response_text_rules: response_json["text_rules"],
                    //         decision_tree_response_accuracy_score: response_json["accuracy_score"]
                    //     })
//

                    // })
                }
            })
    }

    onShowRulesButtonClick = () => {
        this.openRulesModal()
    }

    onDecisionTreeRadioButtonChange = (decision_tree_training_mode) => {
        this.setState({decision_tree_training_mode: decision_tree_training_mode})
    }

    on_decision_tree_max_depth_change = (event) => {
        this.setState({decision_tree_max_depth: event.target.value})
    }

    on_decision_tree_min_samples_split_change = (event) => {
        this.setState({decision_tree_min_samples_split: event.target.value})
    }

    on_decision_tree_decision_tree_max_leaf_nodes_change = (event) => {
        this.setState({decision_tree_max_leaf_nodes: event.target.value})
    }

    onShowUnknownOnlySwitchChange = () => {
        const new_show_unknown_only_value = !this.state.show_unknown_only
        this.setState({show_unknown_only: new_show_unknown_only_value})

        if(new_show_unknown_only_value === true){
            console.log("ToDo update the current image to show only the unknown classes")
        } else {
            console.log("ToDo update the current image to display all the classes")
        }
    }

    onAgglomerativeClusteringRunButtonClick = (agglomerative_clustering_value) => {
        console.log("ToDo run agglomerative clustering with " + agglomerative_clustering_value + " clusters fusion")
        fireSwalError("Not implemented yet!")
    }

    onAgglomerativeClusteringUpdateRulesButtonClick = () => {
        fireSwalError("Not implemented yet!")
        console.log("ToDo update the rules based on the result of the agglomerative clustering")
    }

    onRunModelButtonClick = () => {
        // Sanity checks:
        if(this.state.formatted_features == null){
            fireSwalError("Please load a dataset to visualize")
            return
        }
        if(this.state.selected_class_feature == null){
            fireSwalError("Please select a target feature")
            return
        }
        if(this.state.show_unknown_only === true && this.getUnknownClassesFormattedList().length === 0){
            fireSwalError("No unknown classes to plot", "Try unchecking \"Show unknown classes only\"")
            return
        }
        if(this.getUnknownClassesFormattedList().length === 0){
            fireSwalError("Cannot run clustering", "There are no unknown classes selected")
            return
        }
        if(this.state.model_params_selected_model === "tabularncd"){
            fireSwalError("Not implemented yet!")
            return
        }

        let model_config = null

        if(this.state.model_params_selected_model === "k_means"){
            model_config = {
                'model_name': this.state.model_params_selected_model,

                'k_means_n_clusters': parseInt(this.state.model_params_k_means_n_clusters),
            }
        }

        if(this.state.model_params_selected_model === "spectral_clustering"){
            model_config = {
                'model_name': this.state.model_params_selected_model,

                'spectral_clustering_n_clusters': parseInt(this.state.model_params_spectral_clustering_n_clusters),
                'spectral_clustering_affinity': this.state.model_params_spectral_clustering_affinity,
            }
        }

        if(this.state.model_params_selected_model === "tabularncd"){
            model_config = {
                'model_name': this.state.model_params_selected_model,

                'tabncd_n_clusters': parseInt(this.state.model_params_tabncd_n_clusters),
                'tabncd_cosine_topk': parseFloat(this.state.model_params_tabncd_cosine_topk),
                'tabncd_w1': parseFloat(this.state.model_params_tabncd_w1),
                'tabncd_w2': parseFloat(this.state.model_params_tabncd_w2),
                'tabncd_classifier_lr': parseFloat(this.state.model_params_tabncd_classifier_lr),
                'tabncd_cluster_lr': parseFloat(this.state.model_params_tabncd_cluster_lr),
                'tabncd_k_neighbors': parseInt(this.state.model_params_tabncd_k_neighbors),
                'tabncd_dropout': parseFloat(this.state.model_params_tabncd_dropout),
                'tabncd_activation_fct': this.state.model_params_tabncd_activation_fct
            }
        }

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
        fetch('/runClustering', requestOptions)   // Don't need to specify the full localhost:5000/... as the proxy is set in package.json
            .then(serverPromise => {
                if (serverPromise.status === 500) {
                    fireSwalError('Status 500 - Server error', 'Please make sure that the server is running')
                }
                if (serverPromise.status === 422) {
                    serverPromise.json().then(error => {
                        fireSwalError('Status 422 - Server error', error['error_message'])
                    })
                }
                if (serverPromise.status === 200) {
                    serverPromise.blob().then(image_response_blob => {
                        const imageObjectURL = URL.createObjectURL(image_response_blob);
                        this.setState({image_to_display: imageObjectURL})
                    })
                }
            })
    }

    onAutoParamsButtonClick = () => {
        console.log("ToDo auto params for " + this.state.model_params_selected_model)
        fireSwalError("Not implemented yet!")
    }

    updateSelectedModel = (model_name) => {
        this.setState({ model_params_selected_model: model_name })
    }

    on_kmeans_n_clusters_change = (event) => {
        this.setState({model_params_k_means_n_clusters: event.target.value})
    }

    on_tabncd_n_clusters_change = (event) => {
        this.setState({model_params_tabncd_n_clusters: event.target.value})
    }

    on_tabncd_cosine_topk_change = (event) => {
        this.setState({model_params_tabncd_cosine_topk: event.target.value})
    }

    on_tabncd_w1_change = (event) => {
        this.setState({model_params_tabncd_w1: event.target.value})
    }

    on_tabncd_w2_change = (event) => {
        this.setState({model_params_tabncd_w2: event.target.value})
    }

    on_tabncd_classifier_lr_change = (event) => {
        this.setState({model_params_tabncd_classifier_lr: event.target.value})
    }

    on_tabncd_cluster_lr_change = (event) => {
        this.setState({model_params_tabncd_cluster_lr: event.target.value})
    }

    on_tabncd_k_neighbors_change = (event) => {
        this.setState({model_params_tabncd_k_neighbors: event.target.value})
    }

    on_tabncd_dropout_change = (event) => {
        this.setState({model_params_tabncd_dropout: event.target.value})
    }

    on_tabncd_activation_fct_change = (event) => {
        this.setState({model_params_tabncd_activation_fct: event.target.value})
    }

    on_spectral_clustering_n_clusters_change = (event) => {
        this.setState({model_params_spectral_clustering_n_clusters: event.target.value})
    }

    on_spectral_clustering_affinity_change = (event) => {
        this.setState({model_params_spectral_clustering_affinity: event.target.value})
    }

    onRulesUnknownClassesOnlySwitchChange = () => {
        this.setState({decision_tree_unknown_classes_only: !this.state.decision_tree_unknown_classes_only})
    }

    openRulesModal = () => {
        if(this.state.decision_tree_response_pdf_file === null){
            fireSwalError("No rules to show")
        } else {
            window.open(this.state.decision_tree_response_pdf_file)
        }
        // this.setState({rules_modal_is_open: true})
    }

    closeRulesModal = () => {
        this.setState({rules_modal_is_open: false})
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
            fireSwalError("No image to save.")
        }
    }

    render() {
        return (
            <Row style={{height: '100vh', width:"99vw"}} className="d-flex flex-row justify-content-center align-items-center">

                <RulesDisplayModal rules_modal_is_open={this.state.rules_modal_is_open}
                                   openRulesModal={this.openRulesModal}
                                   closeRulesModal={this.closeRulesModal}
                                   decision_tree_response_training_mode={this.state.decision_tree_response_training_mode}
                                   decision_tree_response_text_rules={this.state.decision_tree_response_text_rules}
                                   decision_tree_response_accuracy_score={this.state.decision_tree_response_accuracy_score}
                                   decision_tree_response_pdf_file={this.state.decision_tree_response_pdf_file}
                />

                <Col className="col-lg-3 col-12 d-flex flex-column" style={{height: "99vh"}}>
                    <Row className="my_row py-2">
                        <DatasetSelector onNewFeaturesLoaded={this.onNewFeaturesLoaded}
                                         setDatasetNameHandler={this.setDatasetNameHandler}
                                         unloadDatasetHandler={this.unloadDatasetHandler}
                        />
                    </Row>
                    <Row className="my_row py-2" style={{flexGrow:'1'}}>
                        <FeatureSelection feature_search_query={this.state.feature_search_query}
                                          onChangeFeaturesSearch={this.onChangeFeaturesSearch}
                                          search_filtered_features_list={this.state.search_filtered_features_list}

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
                        />
                    </Row>
                </Col>

                <Col className="col-lg-6 col-12 d-flex flex-column justify-content-center" style={{height: "99vh"}}>
                    <Row className="my_row mx-lg-1 py-2 d-flex flex-row" style={{flexGrow:'1', height:"100%"}}>
                        <DataVisualization image_to_display={this.state.image_to_display}
                                           onRawDataButtonClick={this.onRawDataButtonClick}
                                           onProjectionButtonClick={this.onProjectionButtonClick}

                                           onShowUnknownOnlySwitchChange={this.onShowUnknownOnlySwitchChange}
                                           show_unknown_only={this.state.show_unknown_only}

                                           onSaveImageButtonClick={this.onSaveImageButtonClick}
                        />
                    </Row>
                </Col>

                <Col className="col-lg-3 col-12 d-flex flex-column" style={{height: "99vh"}}>
                    <Row className="my_row py-2 d-flex flex-row" style={{flexGrow:'1'}}>
                        <ModelSelection onRunModelButtonClick={this.onRunModelButtonClick}
                                        onAutoParamsButtonClick={this.onAutoParamsButtonClick}
                                        updateSelectedModel={this.updateSelectedModel}
                                        model_params_selected_model={this.state.model_params_selected_model}

                                        on_tabncd_n_clusters_change={this.on_tabncd_n_clusters_change}
                                        tabncd_n_clusters={this.state.model_params_tabncd_n_clusters}
                                        on_tabncd_cosine_topk_change={this.on_tabncd_cosine_topk_change}
                                        tabncd_cosine_topk={this.state.model_params_tabncd_cosine_topk}
                                        on_tabncd_w1_change={this.on_tabncd_w1_change}
                                        tabncd_w1={this.state.model_params_tabncd_w1}
                                        on_tabncd_w2_change={this.on_tabncd_w2_change}
                                        tabncd_w2={this.state.model_params_tabncd_w2}
                                        on_tabncd_classifier_lr_change={this.on_tabncd_classifier_lr_change}
                                        tabncd_classifier_lr={this.state.model_params_tabncd_classifier_lr}
                                        on_tabncd_cluster_lr_change={this.on_tabncd_cluster_lr_change}
                                        tabncd_cluster_lr={this.state.model_params_tabncd_cluster_lr}
                                        on_tabncd_k_neighbors_change={this.on_tabncd_k_neighbors_change}
                                        tabncd_k_neighbors={this.state.model_params_tabncd_k_neighbors}
                                        on_tabncd_dropout_change={this.on_tabncd_dropout_change}
                                        tabncd_dropout={this.state.model_params_tabncd_dropout}
                                        on_tabncd_activation_fct_change={this.on_tabncd_activation_fct_change}
                                        tabncd_activation_fct={this.state.model_params_tabncd_activation_fct}

                                        on_kmeans_n_clusters_change={this.on_kmeans_n_clusters_change}
                                        k_means_n_clusters={this.state.model_params_k_means_n_clusters}
                                        onKMeansTrainOnUnknownClassesOnlySwitchChange={this.onKMeansTrainOnUnknownClassesOnlySwitchChange}

                                        on_spectral_clustering_n_clusters_change={this.on_spectral_clustering_n_clusters_change}
                                        spectral_clustering_n_clusters={this.state.model_params_spectral_clustering_n_clusters}
                                        on_spectral_clustering_affinity_change={this.on_spectral_clustering_affinity_change}
                                        spectral_clustering_affinity={this.state.model_params_spectral_clustering_affinity}
                        />
                    </Row>
                    <Row className="my_row py-2 d-flex flex-row">
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
                    <Row className="my_row py-2 d-flex flex-row">
                        <AgglomerativeClustering onAgglomerativeClusteringRunButtonClick={this.onAgglomerativeClusteringRunButtonClick}
                                                 onAgglomerativeClusteringUpdateRulesButtonClick={this.onAgglomerativeClusteringUpdateRulesButtonClick}/>
                    </Row>
                </Col>

            </Row>
        )
    }
}

export default FullPage;