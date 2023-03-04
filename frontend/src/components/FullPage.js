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
            show_model_prediction: false,

            decision_tree_training_mode: "multi_class",
            decision_tree_max_depth: null,
            decision_tree_min_samples_split: 2,
        };

        this.state = this.initial_state;
    }

    onNewFeaturesLoaded = (new_features) => {
        const new_formatted_features = new_features.map((feature, index) => ({"name": feature, "checked" : true, index : index, "disabled": false}))
        this.setState({formatted_features: new_formatted_features})
        this.setState({search_filtered_features_list: this.getUpdatedFilteredList(new_formatted_features, this.state.feature_search_query)})
    }

    onChangeFeaturesSearch = query => {
        if (query.target.value !== '') {
            this.setState({ feature_search_query: query.target.value });
        }

        const updated_filtered_list = this.getUpdatedFilteredList(this.state.formatted_features, query.target.value)
        this.setState({ search_filtered_features_list: updated_filtered_list })
    };

    onChangeUniqueValuesSearch = query => {
        if (query.target.value !== '') {
            this.setState({ unique_values_search_query: query.target.value });
        }

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

    onRawDataButtonClick = () => {
        if(this.state.formatted_features == null){
            fireSwalError("Please load a dataset to visualize")
        } else {
            if(this.state.selected_class_feature == null){
                fireSwalError("Please select a target feature")
            } else {
                const selected_features = []
                this.state.formatted_features.forEach(feature => {
                    if(feature.checked === true && feature.name !== this.state.selected_class_feature){
                        selected_features.push(feature.name)
                    }
                })

                const known_classes = []
                this.state.class_values_to_display.forEach(unique_value => {
                    if(unique_value.checked === true){
                        known_classes.push(unique_value.name)
                    }
                })

                const unknown_classes = []
                this.state.class_values_to_display.forEach(unique_value => {
                    if(unique_value.checked === false){
                        unknown_classes.push(unique_value.name)
                    }
                })

                const requestOptions = {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        'dataset_name': this.state.dataset_name,
                        'selected_features': selected_features,
                        'target_name': this.state.selected_class_feature,
                        'tsne_seed' : 0,
                        'tsne_perplexity': 30.0,
                        'known_classes': known_classes,
                        'unknown_classes': unknown_classes,
                        'color_by': 'known_only'
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
        }
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
        fireSwalError("Not implemented yet!")
        console.log("ToDo run the generation of the rules")
    }

    onShowRulesButtonClick = () => {
        fireSwalError("Not implemented yet!")
        console.log("ToDo display the generated rules")
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

    onShowModelPredictionSwitchChange = () => {
        const new_show_model_prediction_value = !this.state.show_model_prediction
        this.setState({show_model_prediction: new_show_model_prediction_value})

        if(new_show_model_prediction_value === true){
            console.log("ToDo update the current image to include the prediction of the model")
        } else {
            console.log("ToDo update the current image to display only the known classes, the rest has legend \"Unknown\"")
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

    render() {
        return (
            <Row style={{height: '100vh', width:"90vw"}} className="d-flex flex-row justify-content-center align-items-center">

                <Col className="col-lg-3 col-12 d-flex flex-column" style={{height: "95vh"}}>
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

                <Col className="col-lg-6 col-12 d-flex flex-column justify-content-center" style={{height: "95vh"}}>
                    <Row className="my_row mx-lg-1 py-2 d-flex flex-row" style={{flexGrow:'1', height:"100%"}}>
                        <DataVisualization image_to_display={this.state.image_to_display}
                                           onRawDataButtonClick={this.onRawDataButtonClick}
                                           onProjectionButtonClick={this.onProjectionButtonClick}

                                           onShowModelPredictionSwitchChange={this.onShowModelPredictionSwitchChange}
                                           show_model_prediction={this.state.show_model_prediction}
                        />
                    </Row>
                </Col>

                <Col className="col-lg-3 col-12 d-flex flex-column" style={{height: "95vh"}}>
                    <Row className="my_row py-2 d-flex flex-row" style={{flexGrow:'1'}}>
                        <ModelSelection />
                    </Row>
                    <Row className="my_row py-2 d-flex flex-row">
                        <RulesGenerator onDecisionTreeRadioButtonChange={this.onDecisionTreeRadioButtonChange}
                                        decision_tree_training_mode={this.state.decision_tree_training_mode}

                                        on_decision_tree_max_depth_change={this.on_decision_tree_max_depth_change}
                                        decision_tree_max_depth={this.state.decision_tree_max_depth}

                                        on_decision_tree_min_samples_split_change={this.on_decision_tree_min_samples_split_change}
                                        decision_tree_min_samples_split={this.state.decision_tree_min_samples_split}

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