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

            <Row className="d-flex flex-row" style={{marginBottom: "10px"}}>
                <Col className="col-8 d-flex flex-column">
                    <Tooltip title="ToDo tooltip">
                        <div style={{display: "flex", alignItems: "center"}}>
                            Learning rate <FontAwesomeIcon icon={regular('circle-question')} style={{marginLeft: "5px"}}/>
                        </div>
                    </Tooltip>
                </Col>
                <Col className="col-4 d-flex flex-column">
                    <input type="number"
                           min={0}
                           max={1}
                           placeholder="learning rate"
                           step={0.001}
                           onChange={props.on_projection_in_classifier_lr_change}
                           defaultValue={props.projection_in_classifier_lr_value}
                    />
                </Col>
            </Row>

            <Row className="d-flex flex-row" style={{marginBottom: "10px"}}>
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

            <Row className="d-flex flex-row">
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
                                        <th></th>
                                    </tr>
                                    </thead>
                                    <tbody>
                                    <tr>
                                        <td scope="row">Input</td>
                                        <td>
                                            <Tooltip title="(number of selected features, size first hidden layer)">
                                                <div>
                                                    ({format_input_output_size(props.projection_in_classifier_input_size)})
                                                </div>
                                            </Tooltip>
                                        </td>
                                        <td></td>
                                        <td></td>
                                    </tr>
                                    {props.projection_in_classifier_hidden_layers.map((layer_size, layer_index) => (
                                        <tr key={"hidden_layer_row_" + layer_index}>
                                            <td scope="row">Dense {layer_index}</td>
                                            <td>({(layer_index === 0)
                                                    ? format_input_output_size(props.projection_in_classifier_input_size)
                                                    : props.projection_in_classifier_hidden_layers[layer_index - 1]}, {layer_size})</td>
                                            <td>{(layer_index === 0)
                                                ? (props.projection_in_classifier_input_size === null)
                                                    ? '?'
                                                    : props.projection_in_classifier_input_size * layer_size + layer_size
                                                : props.projection_in_classifier_hidden_layers[layer_index - 1] * layer_size + layer_size}</td>
                                            <td>
                                                <Button className="btn-secondary"
                                                        style={{borderRadius: "50px", padding: "3px", paddingRight: "2px", paddingBottom: "1px", paddingTop: "1px"}}
                                                        onClick={() => props.onProjectionInClassifierRemoveLayerButtonClick(layer_index)}
                                                >
                                                    <div className="d-flex align-items-center">
                                                        <FontAwesomeIcon icon={solid('minus')}/>
                                                    </div>
                                                </Button>
                                            </td>
                                        </tr>
                                    ))}
                                    <tr>
                                        <td colSpan="4">
                                            <center>
                                                Add layer:
                                                <input id="projectionInClassifierLayerSizeInput"
                                                       type="number"
                                                       min={0}
                                                       placeholder="size"
                                                       step={1}
                                                       style={{marginLeft: "5px", marginRight: "5px", maxWidth: "60px"}}
                                                />
                                                <Button className="btn-secondary"
                                                        style={{borderRadius: "50px", padding: "3px", paddingRight: "2px", paddingBottom: "1px", paddingTop: "1px"}}
                                                        onClick={props.onProjectionInClassifierAddLayerButtonClick}
                                                >
                                                    <div className="d-flex align-items-center">
                                                        <FontAwesomeIcon icon={solid('plus')}/>
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
                                                    ({(props.projection_in_classifier_hidden_layers.length > 0)
                                                    ? props.projection_in_classifier_hidden_layers[props.projection_in_classifier_hidden_layers.length - 1]
                                                    : format_input_output_size(props.projection_in_classifier_input_size)}, {format_input_output_size(props.projection_in_classifier_output_size)})
                                                </div>
                                            </Tooltip>
                                        </td>
                                        <td>{(props.projection_in_classifier_hidden_layers.length > 0)
                                            ? (props.projection_in_classifier_output_size !== null)
                                                ? props.projection_in_classifier_hidden_layers[props.projection_in_classifier_hidden_layers.length - 1] * props.projection_in_classifier_output_size + props.projection_in_classifier_output_size
                                                : '?'
                                            : ((props.projection_in_classifier_input_size === null || props.projection_in_classifier_output_size === null)
                                                ? '?'
                                                : props.projection_in_classifier_input_size * props.projection_in_classifier_output_size + props.projection_in_classifier_output_size)}</td>
                                        <td></td>
                                    </tr>
                                    </tbody>
                                </table>
                            </div>
                        </Row>
                    </Col>
                </div>
            </Row>
        </Container>
    )

}

export default ProjectionInClassifierParameters;