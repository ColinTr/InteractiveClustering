import React from "react";
import Container from "react-bootstrap/Container";
import Col from "react-bootstrap/Col";
import Row from "react-bootstrap/Row";
import { Tooltip } from "@mui/material";
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome'
import { regular, solid } from '@fortawesome/fontawesome-svg-core/import.macro'
import Form from "react-bootstrap/Form";
import Button from "react-bootstrap/Button";
import {styled} from "@mui/material/styles";
import {tooltipClasses} from "@mui/material/Tooltip";


function format_input_output_size(size) {
    if(size === null) {
        return "?"
    } else {
        return size
    }
}

const NoMaxWidthTooltip = styled(({ className, ...props }) => (
    <Tooltip {...props} classes={{ popper: className }} />
))({
    [`& .${tooltipClasses.tooltip}`]: {
        maxWidth: 'none',
    },
});

const ProjectionInClassifierParameters = (props) => {
    return (
        <Container>
            <Row className="d-flex flex-row" style={{marginBottom: "10px"}}>
                <Col className="col-auto">
                    <NoMaxWidthTooltip title={<img src="/ProjectionInClassifier_architecture.png" alt="Model architecture" width="600"/>} placement="bottom-end">
                        <u>Model architecture help</u>
                    </NoMaxWidthTooltip >
                </Col>
                /
                <Col className="col-auto">
                    <a rel="noreferrer" target="_blank" href='https://arxiv.org/pdf/2209.01217' style={{color: "white"}}>Article</a>
                </Col>
            </Row>

            <Row className="d-flex flex-row" style={{marginBottom: "10px"}}>
                <Col className="col-8 d-flex flex-column">
                    <Tooltip title="The number of clusters used in the k-means after projection">
                        <div style={{display: "flex", alignItems: "center"}}>
                            Number of clusters <FontAwesomeIcon icon={regular('circle-question')} style={{marginLeft: "5px"}}/>
                        </div>
                    </Tooltip>
                </Col>
                <Col className="col-4 d-flex flex-column">
                    <input type="number"
                           min={0}
                           placeholder="n clusters"
                           step={1}
                           onChange={props.on_projection_in_classifier_n_clusters_change}
                           defaultValue={props.projection_in_classifier_n_clusters}
                    />
                </Col>
            </Row>

            <Row className="d-flex flex-row" style={{marginBottom: "10px"}}>
                <Col className="col-8 d-flex flex-column">
                    <Tooltip title="The probability of dropping neurons during training for regularization">
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
                    <Tooltip title="The step size at each iteration while moving toward the minimum of the loss function">
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
                <Col className="col-8 d-flex flex-column">
                    <Tooltip title="The number of times that the training will go over the entire dataset">
                        <div style={{display: "flex", alignItems: "center"}}>
                            Number of epochs <FontAwesomeIcon icon={regular('circle-question')} style={{marginLeft: "5px"}}/>
                        </div>
                    </Tooltip>
                </Col>
                <Col className="col-4 d-flex flex-column">
                    <input type="number"
                           min={0}
                           placeholder="epochs"
                           step={1}
                           onChange={props.on_projection_in_classifier_epochs_change}
                           defaultValue={props.projection_in_classifier_epochs}
                    />
                </Col>
            </Row>

            <Row className="d-flex flex-row" style={{marginBottom: "10px"}}>
                <Col className="col-7 d-flex flex-column">
                    <Tooltip title="The activation function used between the hidden layers">
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
                        <option value="relu">ReLu</option>
                        <option value="sigmoid">Sigmoid</option>
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
                                            <Tooltip title="(number of selected features)">
                                                <div>
                                                    ({format_input_output_size(props.n_features_used)})
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
                                                    ? format_input_output_size(props.n_features_used)
                                                    : props.projection_in_classifier_hidden_layers[layer_index - 1]}, {layer_size})</td>
                                            <td>{(layer_index === 0)
                                                ? (props.n_features_used === null)
                                                    ? '?'
                                                    : props.n_features_used * layer_size + layer_size
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
                                                    : format_input_output_size(props.n_features_used)}, {format_input_output_size(props.n_known_classes)})
                                                </div>
                                            </Tooltip>
                                        </td>
                                        <td>{(props.projection_in_classifier_hidden_layers.length > 0)
                                            ? (props.n_known_classes !== null)
                                                ? props.projection_in_classifier_hidden_layers[props.projection_in_classifier_hidden_layers.length - 1] * props.n_known_classes + props.n_known_classes
                                                : '?'
                                            : ((props.n_features_used === null || props.n_known_classes === null)
                                                ? '?'
                                                : props.n_features_used * props.n_known_classes + props.n_known_classes)}</td>
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