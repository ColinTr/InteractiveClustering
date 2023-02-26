import React from "react";
import Container from "react-bootstrap/Container";
import Row from "react-bootstrap/Row";
import Col from "react-bootstrap/Col";
import Form from "react-bootstrap/Form";


const TabularNCDParameters = (props) => {
    return (
        <Container>
            <Row className="d-flex flex-row">
                <Col className="col-8 d-flex flex-column">
                    <p>Number of clusters</p>
                </Col>
                <Col className="col-4 d-flex flex-column">
                    <input type="number"
                           min={0}
                           placeholder="Number of clusters"
                           step={1}
                           onChange={props.on_tabncd_n_clusters_change}
                           defaultValue={props.tabncd_n_clusters_value}
                    />
                </Col>
            </Row>

            <Row className="d-flex flex-row">
                <Col className="col-8 d-flex flex-column">
                    <p>Cosine Top <i>k</i> (in %)</p>
                </Col>
                <Col className="col-4 d-flex flex-column">
                    <input type="number"
                           min={0}
                           max={1}
                           placeholder="Cosine top k"
                           step={0.1}
                           onChange={props.on_tabncd_cosine_topk_change}
                           defaultValue={props.tabncd_cosine_topk_value}
                    />
                </Col>
            </Row>

            <Row className="d-flex flex-row">
                <Col className="col-8 d-flex flex-column">
                    <p>w1</p>
                </Col>
                <Col className="col-4 d-flex flex-column">
                    <input type="number"
                           min={0}
                           max={1}
                           placeholder="w1"
                           step={0.1}
                           onChange={props.on_tabncd_w1_change}
                           defaultValue={props.tabncd_w1_value}
                    />
                </Col>
            </Row>

            <Row className="d-flex flex-row">
                <Col className="col-8 d-flex flex-column">
                    <p>w2</p>
                </Col>
                <Col className="col-4 d-flex flex-column">
                    <input type="number"
                           min={0}
                           max={1}
                           placeholder="w2"
                           step={0.1}
                           onChange={props.on_tabncd_w2_change}
                           defaultValue={props.tabncd_w2_value}
                    />
                </Col>
            </Row>

            <Row className="d-flex flex-row">
                <Col className="col-8 d-flex flex-column">
                    <p>Classif. lr</p>
                </Col>
                <Col className="col-4 d-flex flex-column">
                    <input type="number"
                           min={0}
                           max={1}
                           placeholder="classif. lr"
                           step={0.001}
                           onChange={props.on_tabncd_classifier_lr_change}
                           defaultValue={props.tabncd_classifier_lr_value}
                    />
                </Col>
            </Row>

            <Row className="d-flex flex-row">
                <Col className="col-8 d-flex flex-column">
                    <p>Clust. lr</p>
                </Col>
                <Col className="col-4 d-flex flex-column">
                    <input type="number"
                           min={0}
                           max={1}
                           placeholder="classif. lr"
                           step={0.001}
                           onChange={props.on_tabncd_cluster_lr_change}
                           defaultValue={props.tabncd_cluster_lr_value}
                    />
                </Col>
            </Row>

            <Row className="d-flex flex-row">
                <Col className="col-8 d-flex flex-column">
                    <p><i>k</i> neighbors</p>
                </Col>
                <Col className="col-4 d-flex flex-column">
                    <input type="number"
                           min={0}
                           placeholder="k neighbors"
                           step={1}
                           onChange={props.on_tabncd_k_neighbors_change}
                           defaultValue={props.tabncd_k_neighbors_value}
                    />
                </Col>
            </Row>

            <Row className="d-flex flex-row">
                <Col className="col-8 d-flex flex-column">
                    <p>Dropout</p>
                </Col>
                <Col className="col-4 d-flex flex-column">
                    <input type="number"
                           min={0}
                           max={1}
                           placeholder="dropout"
                           step={0.1}
                           onChange={props.on_tabncd_dropout_change}
                           defaultValue={props.tabncd_dropout_value}
                    />
                </Col>
            </Row>

            <Row className="d-flex flex-row">
                <Col className="col-7 d-flex flex-column">
                    <p>Activation function</p>
                </Col>
                <Col className="col-5 d-flex flex-column">
                    <Form.Select
                        aria-label="Activation function"
                        onChange={props.on_tabncd_activation_fct_change}
                        style={{paddingTop: 0, paddingLeft: "3px", paddingBottom: 0}}
                    >
                        <option value="sigmoid">Sigmoid</option>
                        <option value="relu">ReLu</option>
                        <option value="non">None</option>
                    </Form.Select>
                </Col>
            </Row>
        </Container>
    )
}

export default TabularNCDParameters;