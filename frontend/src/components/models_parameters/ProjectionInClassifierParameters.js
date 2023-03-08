import React from "react";
import Container from "react-bootstrap/Container";
import Col from "react-bootstrap/Col";
import Row from "react-bootstrap/Row";
import { Tooltip } from "@mui/material";
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome'
import { regular, solid } from '@fortawesome/fontawesome-svg-core/import.macro'
import Form from "react-bootstrap/Form";
import Button from "react-bootstrap/Button";


function format_input_output_size(size) {
    if(size === null) {
        return "?"
    } else {
        return size
    }
}


const ProjectionInClassifierParameters = (props) => {
    return (
        <Container>
            <Row className="d-flex flex-row" style={{marginBottom: "10px"}}>
                <Col className="col-8 d-flex flex-column">
                    <Tooltip title="The number of clusters used in the k-means after projection.">
                        <div style={{display: "flex", alignItems: "center"}}>
                            Number of clusters <FontAwesomeIcon icon={regular('circle-question')} style={{marginLeft: "5px"}}/>
                        </div>
                    </Tooltip>
                </Col>
                <Col className="col-4 d-flex flex-column">
                    <input type="number"
                           min={0}
                           placeholder="Number of clusters"
                           step={1}
                           onChange={props.on_projection_in_classifier_n_clusters_change}
                           defaultValue={props.projection_in_classifier_n_clusters}
                    />
                </Col>
            </Row>

            <Row className="d-flex flex-row" style={{marginBottom: "10px"}}>
                <div>
                    <Col className="d-flex flex-column">
                        <Row>
                            <div>
                                Architecture:
                            </div>
                        </Row>
                        {/* style={{border: "0.5mm solid", borderRadius: "0.375rem", padding: "5px"}} */}
                        <Row>
                            <div>
                                <table className="table" style={{color: "white"}}>
                                    <thead>
                                        <tr>
                                            <th scope="col">Layer</th>
                                            <th scope="col">Shape</th>
                                            <th scope="col">Param #</th>
                                        </tr>
                                    </thead>
                                    <tbody>
                                        <tr>
                                            <td scope="row">Input</td>
                                            <td>
                                                <Tooltip title="(number of selected features, size first hidden layer)">
                                                    <div>
                                                        ({format_input_output_size(props.projection_in_classifier_input_size)}, ?)
                                                    </div>
                                                </Tooltip>
                                            </td>
                                            <td>?</td>
                                        </tr>
                                        <tr>
                                            <td colSpan="4">
                                                <center>
                                                    <Button className="btn-secondary" style={{paddingRight: "5px", paddingLeft: "5px", paddingTop: 0, paddingBottom: 0}}>
                                                        <div className="d-flex align-items-center">
                                                            <FontAwesomeIcon icon={solid('circle-plus')} style={{marginRight: "6px"}}/> Add layer
                                                        </div>
                                                    </Button>
                                                </center>
                                            </td>
                                        </tr>
                                        <tr>
                                            <td scope="row">Output</td>
                                            <td>
                                                <Tooltip title="(size last hidden layer, number of known classes)">
                                                    <div>
                                                        (?, {format_input_output_size(props.projection_in_classifier_output_size)})
                                                    </div>
                                                </Tooltip>
                                            </td>
                                            <td>?</td>
                                        </tr>
                                    </tbody>
                                </table>
                            </div>
                        </Row>
                    </Col>
                </div>
            </Row>

            <Row className="d-flex flex-row" style={{marginBottom: "10px"}}>
                <Col className="col-8 d-flex flex-column">
                    <Tooltip title="ToDo tooltip">
                        <div style={{display: "flex", alignItems: "center"}}>
                            Dropout <FontAwesomeIcon icon={regular('circle-question')} style={{marginLeft: "5px"}}/>
                        </div>
                    </Tooltip>
                </Col>
                <Col className="col-4 d-flex flex-column">
                    <input type="number"
                           min={0}
                           max={1}
                           placeholder="dropout"
                           step={0.1}
                           onChange={props.on_projection_in_classifier_dropout_change}
                           defaultValue={props.projection_in_classifier_dropout}
                    />
                </Col>
            </Row>

            <Row className="d-flex flex-row">
                <Col className="col-7 d-flex flex-column">
                    <Tooltip title="ToDo tooltip">
                        <div style={{display: "flex", alignItems: "center"}}>
                            Activation function <FontAwesomeIcon icon={regular('circle-question')} style={{marginLeft: "5px"}}/>
                        </div>
                    </Tooltip>
                </Col>
                <Col className="col-5 d-flex flex-column">
                    <Form.Select
                        aria-label="Activation function"
                        onChange={props.on_projection_in_classifier_activation_fct_change}
                        style={{paddingTop: 0, paddingLeft: "3px", paddingBottom: 0}}
                    >
                        <option value="sigmoid">Sigmoid</option>
                        <option value="relu">ReLu</option>
                        <option value="none">None</option>
                    </Form.Select>
                </Col>
            </Row>
        </Container>
    )

}

export default ProjectionInClassifierParameters;