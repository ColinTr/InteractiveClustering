/*
Software Name : InteractiveClustering
Version: 1.0
SPDX-FileCopyrightText: Copyright (c) 2024 Orange
SPDX-License-Identifier: MIT
This software is distributed under the MIT License,
the text of which is available at https://spdx.org/licenses/MIT.html
or see the "license.txt" file for more details.
*/

import React from "react";
import Container from "react-bootstrap/Container";
import Form from 'react-bootstrap/Form';
import Button from "react-bootstrap/Button";
import Col from "react-bootstrap/Col";
import Row from "react-bootstrap/Row";

import KMeansParameters from "./models_parameters/KMeansParameters";
import TabularNCDParameters from "./models_parameters/TabularNCDParameters";
import SpectralClusteringParameters from "./models_parameters/SpectralClusteringParameters";
import ProjectionInClassifierParameters from "./models_parameters/ProjectionInClassifierParameters";
import PBNParameters from "./models_parameters/PBNParameters";

import '../App.css';

class ModelSelection extends React.Component {

    constructor(props) {
        super(props);
        this.state = {
            selected_model: "pbn"
        };
    }

    onDropDownChange = event => {
        this.props.updateSelectedModel(event.target.value)
        this.setState({selected_model: event.target.value})
    };

    render() {
        return (
            <Container>
                <Col className="d-flex flex-column" style={{height: '100%', paddingLeft: "6px", paddingRight: "6px"}}>
                    <Row className="d-flex flex-row">
                        <h5>Model selection</h5>
                    </Row>

                    <hr/>

                    <Row className="d-flex flex-row" style={{paddingTop: "0px"}}>
                        <Col className="col-4 d-flex flex-column justify-content-center">
                            Model
                        </Col>
                        <Col className="col-8 d-flex flex-column">
                            <Form.Select aria-label="Default select example" onChange={this.onDropDownChange} className="my-row">
                                <option value="pbn">PBN</option>
                                <option value="tabularncd">TabularNCD</option>
                                <option value="projection_in_classifier">Projection in classifier</option>
                                <option value="k_means">k-means</option>
                                <option value="spectral_clustering">Spectral clustering</option>
                            </Form.Select>
                        </Col>
                    </Row>

                    <hr/>

                    <Row className="d-flex flex-row py-2" style={{overflowY: "auto", flexGrow:'1', flex:"1 1 auto", height: "0px"}}>
                        {this.state.selected_model === "pbn" &&
                            <PBNParameters
                                n_features_used={this.props.n_features_used}
                                n_known_classes={this.props.n_known_classes}

                                on_pbn_epochs_change={this.props.on_pbn_epochs_change}
                                pbn_epochs_value={this.props.pbn_epochs}

                                on_pbn_n_clusters_change={this.props.on_pbn_n_clusters_change}
                                pbn_n_clusters_value={this.props.pbn_n_clusters}

                                on_pbn_w_change={this.props.on_pbn_w_change}
                                pbn_w_value={this.props.pbn_w}

                                on_pbn_lr_change={this.props.on_pbn_lr_change}
                                pbn_lr_value={this.props.pbn_lr}

                                on_pbn_dropout_change={this.props.on_pbn_dropout_change}
                                pbn_dropout_value={this.props.pbn_dropout}

                                on_pbn_activation_fct_change={this.props.on_pbn_activation_fct_change}

                                pbn_hidden_layers={this.props.pbn_hidden_layers}
                                on_pbn_add_layer_button_click={this.props.on_pbn_add_layer_button_click}
                                on_pbn_remove_layer_button_click={this.props.on_pbn_remove_layer_button_click}
                            />
                        }
                        {this.state.selected_model === "tabularncd" &&
                            <TabularNCDParameters
                                n_features_used={this.props.n_features_used}
                                n_known_classes={this.props.n_known_classes}

                                on_tabncd_epochs_change={this.props.on_tabncd_epochs_change}
                                tabncd_epochs_value={this.props.tabncd_epochs}

                                on_tabncd_n_clusters_change={this.props.on_tabncd_n_clusters_change}
                                tabncd_n_clusters_value={this.props.tabncd_n_clusters}

                                on_tabncd_cosine_topk_change={this.props.on_tabncd_cosine_topk_change}
                                tabncd_cosine_topk_value={this.props.tabncd_cosine_topk}

                                on_tabncd_w1_change={this.props.on_tabncd_w1_change}
                                tabncd_w1_value={this.props.tabncd_w1}

                                on_tabncd_w2_change={this.props.on_tabncd_w2_change}
                                tabncd_w2_value={this.props.tabncd_w2}

                                on_tabncd_topk_change={this.props.on_tabncd_topk_change}
                                tabncd_topk_value={this.props.tabncd_topk}

                                on_tabncd_lr_change={this.props.on_tabncd_lr_change}
                                tabncd_lr_value={this.props.tabncd_lr}

                                on_tabncd_k_neighbors_change={this.props.on_tabncd_k_neighbors_change}
                                tabncd_k_neighbors_value={this.props.tabncd_k_neighbors}

                                on_tabncd_dropout_change={this.props.on_tabncd_dropout_change}
                                tabncd_dropout_value={this.props.tabncd_dropout}

                                on_tabncd_activation_fct_change={this.props.on_tabncd_activation_fct_change}

                                tabncd_hidden_layers={this.props.tabncd_hidden_layers}
                                onTabncdAddLayerButtonClick={this.props.onTabncdAddLayerButtonClick}
                                onTabncdRemoveLayerButtonClick={this.props.onTabncdRemoveLayerButtonClick}
                            />
                        }
                        {this.state.selected_model === "k_means" &&
                            <KMeansParameters
                                on_kmeans_n_clusters_change={this.props.on_kmeans_n_clusters_change}
                                k_means_n_clusters={this.props.k_means_n_clusters}
                            />
                        }
                        {this.state.selected_model === "spectral_clustering" &&
                            <SpectralClusteringParameters
                                on_spectral_clustering_n_clusters_change={this.props.on_spectral_clustering_n_clusters_change}
                                spectral_clustering_n_clusters={this.props.spectral_clustering_n_clusters}

                                on_spectral_clustering_affinity_change={this.props.on_spectral_clustering_affinity_change}
                                spectral_clustering_affinity={this.props.spectral_clustering_affinity}
                            />
                        }
                        {this.state.selected_model === "projection_in_classifier" &&
                            <ProjectionInClassifierParameters
                                n_features_used = {this.props.n_features_used}
                                n_known_classes = {this.props.n_known_classes}

                                on_projection_in_classifier_epochs_change = {this.props.on_projection_in_classifier_epochs_change}
                                projection_in_classifier_epochs = {this.props.projection_in_classifier_epochs}
                                on_projection_in_classifier_n_clusters_change = {this.props.on_projection_in_classifier_n_clusters_change}
                                projection_in_classifier_n_clusters = {this.props.projection_in_classifier_n_clusters}
                                projection_in_classifier_hidden_layers = {this.props.projection_in_classifier_hidden_layers}
                                on_projection_in_classifier_dropout_change = {this.props.on_projection_in_classifier_dropout_change}
                                projection_in_classifier_dropout = {this.props.projection_in_classifier_dropout}
                                on_projection_in_classifier_activation_fct_change = {this.props.on_projection_in_classifier_activation_fct_change}
                                on_projection_in_classifier_lr_change = {this.props.on_projection_in_classifier_lr_change}
                                projection_in_classifier_lr_value = {this.props.projection_in_classifier_lr_value}

                                onProjectionInClassifierAddLayerButtonClick = {this.props.onProjectionInClassifierAddLayerButtonClick}
                                onProjectionInClassifierRemoveLayerButtonClick = {this.props.onProjectionInClassifierRemoveLayerButtonClick}
                            />
                        }
                    </Row>

                    <Row className="d-flex flex-row" style={{paddingTop: "10px"}}>
                        <Col className="col-6 d-flex flex-column">
                            <Button style={{width:'120px'}} onClick={() => this.props.onRunModelButtonClick()}>
                                Run
                            </Button>
                        </Col>
                        <Col variant="primary" className="col-6 d-flex flex-column align-items-end">
                            <Button variant="success" style={{width:'120px'}} onClick={() => this.props.onAutoParamsButtonClick()}>
                                Auto params
                            </Button>
                        </Col>
                    </Row>
                </Col>
            </Container>
        )
    }
}

export default ModelSelection;