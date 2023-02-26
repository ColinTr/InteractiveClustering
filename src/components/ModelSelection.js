import React from "react";
import Container from "react-bootstrap/Container";
import Form from 'react-bootstrap/Form';
import Button from "react-bootstrap/Button";
import Col from "react-bootstrap/Col";
import Row from "react-bootstrap/Row";
import KMeansParameters from "./models_parameters/KMeansParameters";
import '../App.css';
import TabularNCDParameters from "./models_parameters/TabularNCDParameters";

class ModelSelection extends React.Component {

    // Default k means parameters :
    default_kmeans_n_clusters = 10

    // Default TabularNCD parameters :
    default_tabncd_n_clusters = 10
    default_tabncd_cosine_topk = 10
    default_tabncd_w1 = 0.8
    default_tabncd_w2 = 0.8
    default_tabncd_classifier_lr = 0.001
    default_tabncd_cluster_lr = 0.001
    default_tabncd_k_neighbors = 5
    default_tabncd_dropout = 0.2
    default_tabncd_activation_fct = "Sigmoid"

    constructor(props) {
        super(props);
        this.state = {
            selected_model : "k_means",

            parameters_to_display : null,

            // k means parameters :
            k_means_n_clusters: this.default_kmeans_n_clusters,

            // TabularNCD parameters :
            tabncd_n_clusters : this.default_tabncd_n_clusters,
            tabncd_cosine_topk : this.default_tabncd_cosine_topk,
            tabncd_w1 : this.default_tabncd_w1,
            tabncd_w2 : this.default_tabncd_w2,
            tabncd_classifier_lr : this.default_tabncd_classifier_lr,
            tabncd_cluster_lr : this.default_tabncd_cluster_lr,
            tabncd_k_neighbors : this.default_tabncd_k_neighbors,
            tabncd_dropout : this.default_tabncd_dropout,
            tabncd_activation_fct : this.default_tabncd_activation_fct,
        };

        this.set_display_to_tabularncd()
    }

    on_kmeans_n_clusters_change = (event) => {
        this.setState({k_means_n_clusters: event.target.value})
    }

    on_tabncd_n_clusters_change = (event) => {
        this.setState({tabncd_n_clusters: event.target.value})
    }

    on_tabncd_cosine_topk_change = (event) => {
        this.setState({tabncd_cosine_topk: event.target.value})
    }

    on_tabncd_w1_change = (event) => {
        this.setState({tabncd_w1: event.target.value})
    }

    on_tabncd_w2_change = (event) => {
        this.setState({tabncd_w2: event.target.value})
    }

    on_tabncd_classifier_lr_change = (event) => {
        this.setState({tabncd_classifier_lr: event.target.value})
    }

    on_tabncd_cluster_lr_change = (event) => {
        this.setState({tabncd_cluster_lr: event.target.value})
    }

    on_tabncd_k_neighbors_change = (event) => {
        this.setState({tabncd_k_neighbors: event.target.value})
    }

    on_tabncd_dropout_change = (event) => {
        this.setState({tabncd_dropout: event.target.value})
    }

    on_tabncd_activation_fct_change = (event) => {
        this.setState({tabncd_activation_fct: event.target.value})
    }

    set_display_to_kmeans = () => {
        this.state.parameters_to_display = <KMeansParameters
            on_kmeans_n_clusters_change={this.on_kmeans_n_clusters_change}
            n_clusters_value={this.state.k_means_n_clusters}
        />
    }

    set_display_to_tabularncd = () => {
        this.state.parameters_to_display = <TabularNCDParameters
            on_tabncd_n_clusters_change={this.on_tabncd_n_clusters_change}
            tabncd_n_clusters_value={this.state.tabncd_n_clusters}
            on_tabncd_cosine_topk_change={this.on_tabncd_cosine_topk_change}
            tabncd_cosine_topk_value={this.state.tabncd_cosine_topk}
            on_tabncd_w1_change={this.on_tabncd_w1_change}
            tabncd_w1_value={this.state.tabncd_w1}
            on_tabncd_w2_change={this.on_tabncd_w2_change}
            tabncd_w2_value={this.state.tabncd_w2}
            on_tabncd_classifier_lr_change={this.on_tabncd_classifier_lr_change}
            tabncd_classifier_lr_value={this.state.tabncd_classifier_lr}
            on_tabncd_cluster_lr_change={this.on_tabncd_cluster_lr_change}
            tabncd_cluster_lr_value={this.state.tabncd_cluster_lr}
            on_tabncd_k_neighbors_change={this.on_tabncd_k_neighbors_change}
            tabncd_k_neighbors_value={this.state.tabncd_k_neighbors}
            on_tabncd_dropout_change={this.on_tabncd_dropout_change}
            tabncd_dropout_value={this.state.tabncd_dropout}
            on_tabncd_activation_fct_change={this.on_tabncd_activation_fct_change}
            tabncd_activation_fct_value={this.state.tabncd_activation_fct}
        />
    }

    onDropDownChange = event => {
        this.setState({ selected_model: event.target.value });

        console.log("Selected model " + event.target.value)

        if (event.target.value === "k_means") {
            this.set_display_to_kmeans()
        }

        if (event.target.value === "tabularncd") {
            this.set_display_to_tabularncd()
        }

        if (event.target.value === "...") {
            this.state.parameters_to_display = <p>Unimplemented model</p>
        }

    };

    onRunButtonClick = () => (
        console.log("ToDo run model " + this.state.selected_model)
    )

    onAutoParamsButtonClick = () => (
        console.log("ToDo auto params for " + this.state.selected_model)
    )

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
                                <option value="tabularncd">TabularNCD</option>
                                <option value="k_means">k-means</option>
                                <option value="...">...</option>
                            </Form.Select>
                        </Col>
                    </Row>

                    <hr/>

                    <Row className="d-flex flex-row py-2" style={{overflowY: "auto", flexGrow:'1', flex:"1 1 auto", height: "0px"}}>
                        {this.state.parameters_to_display}
                    </Row>

                    <Row className="d-flex flex-row" style={{paddingTop: "10px"}}>
                        <Col className="col-6 d-flex flex-column">
                            <Button variant="success" style={{paddingLeft:0, paddingRight:0}} onClick={() => this.onAutoParamsButtonClick()}>
                                Auto params
                            </Button>
                        </Col>
                        <Col variant="primary" className="col-6 d-flex flex-column">
                            <Button style={{paddingLeft:0, paddingRight:0}} onClick={() => this.onRunButtonClick()}>
                                Run
                            </Button>
                        </Col>
                    </Row>
                </Col>
            </Container>
        )
    }
}

export default ModelSelection;