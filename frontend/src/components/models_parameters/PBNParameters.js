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
import Row from "react-bootstrap/Row";
import Col from "react-bootstrap/Col";
import Form from "react-bootstrap/Form";
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome'
import {regular, solid} from '@fortawesome/fontawesome-svg-core/import.macro'
import Button from "react-bootstrap/Button";
import { styled } from '@mui/material/styles';
import Tooltip, { tooltipClasses } from '@mui/material/Tooltip';


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

const PBNParameters = (props) => {
    return (
        <Container>
            <Row className="d-flex flex-row" style={{marginBottom: "10px"}}>
                <Col className="col-auto">
                    <NoMaxWidthTooltip title={<img src="/PBN_architecture.png" alt="Model architecture" width="1100"/>} placement="bottom-end">
                        <u>Model architecture help</u>
                    </NoMaxWidthTooltip >
                </Col>
                /
                <Col className="col-auto">
                    <a rel="noreferrer" target="_blank" href='https://arxiv.org/pdf/2311.05440' style={{color: "white"}}>Article</a>
                </Col>
            </Row>

            <Row className="d-flex flex-row" style={{marginBottom: "10px"}}>
                <Col className="col-8 d-flex flex-column">
                    <Tooltip title="The number of clusters to form">
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
                           onChange={props.on_pbn_n_clusters_change}
                           value={props.pbn_n_clusters_value}
                    />
                </Col>
            </Row>

            <Row className="d-flex flex-row" style={{marginBottom: "10px"}}>
                <Col className="col-8 d-flex flex-column">
                    <Tooltip title="The trade-off parameter">
                        <div style={{display: "flex", alignItems: "center"}}>
                            w <FontAwesomeIcon icon={regular('circle-question')} style={{marginLeft: "5px"}}/>
                        </div>
                    </Tooltip>
                </Col>
                <Col className="col-4 d-flex flex-column">
                    <input type="number"
                           min={0}
                           max={1}
                           placeholder="w"
                           step={0.1}
                           onChange={props.on_pbn_w_change}
                           value={props.pbn_w_value}
                    />
                </Col>
            </Row>

            <Row className="d-flex flex-row" style={{marginBottom: "10px"}}>
                <Col className="col-8 d-flex flex-column">
                    <Tooltip title="The step size at each iteration of the loss of the neural network">
                        <div style={{display: "flex", alignItems: "center"}}>
                            learning rate <FontAwesomeIcon icon={regular('circle-question')} style={{marginLeft: "5px"}}/>
                        </div>
                    </Tooltip>
                </Col>
                <Col className="col-4 d-flex flex-column">
                    <input type="number"
                           min={0}
                           max={1}
                           placeholder="lr"
                           step={0.001}
                           onChange={props.on_pbn_lr_change}
                           value={props.pbn_lr_value}
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
                           onChange={props.on_pbn_epochs_change}
                           value={props.pbn_epochs_value}
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
                           onChange={props.on_pbn_dropout_change}
                           value={props.pbn_dropout_value}
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
                        onChange={props.on_pbn_activation_fct_change}
                        value={props.pbn_activation_fct}
                        style={{paddingTop: 0, paddingLeft: "3px", paddingBottom: 0}}
                    >
                        <option value="relu">ReLu</option>
                        <option value="sigmoid">Sigmoid</option>
                        <option value="none">None</option>
                    </Form.Select>
                </Col>
            </Row>

            <Row className="d-flex flex-row" style={{marginBottom: "10px"}}>
                <div>
                    <Col className="d-flex flex-column">
                        <Row>
                            <div>
                                Encoder's architecture:
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
                                    {props.pbn_hidden_layers.map((layer_size, layer_index) => (
                                        <tr key={"hidden_layer_row_" + layer_index}>
                                            <td scope="row">Dense {layer_index}</td>
                                            <td>({(layer_index === 0)
                                                ? format_input_output_size(props.n_features_used)
                                                : props.pbn_hidden_layers[layer_index - 1]}, {layer_size})</td>
                                            <td>{(layer_index === 0)
                                                ? (props.n_features_used === null)
                                                    ? '?'
                                                    : props.n_features_used * layer_size + layer_size
                                                : props.pbn_hidden_layers[layer_index - 1] * layer_size + layer_size}</td>
                                            <td>
                                                <Button className="btn-secondary"
                                                        style={{borderRadius: "50px", padding: "3px", paddingRight: "2px", paddingBottom: "1px", paddingTop: "1px"}}
                                                        onClick={() => props.on_pbn_remove_layer_button_click(layer_index)}
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
                                                    <input id="pbnLayerSizeInput"
                                                           type="number"
                                                           min={0}
                                                           placeholder="size"
                                                           step={1}
                                                           style={{marginLeft: "5px", marginRight: "5px", maxWidth: "60px"}}
                                                    />
                                                    <Button className="btn-secondary"
                                                            style={{borderRadius: "50px", padding: "3px", paddingRight: "2px", paddingBottom: "1px", paddingTop: "1px"}}
                                                            onClick={props.on_pbn_add_layer_button_click}
                                                    >
                                                    <div className="d-flex align-items-center">
                                                        <FontAwesomeIcon icon={solid('plus')}/>
                                                    </div>
                                                </Button>
                                            </center>
                                        </td>
                                    </tr>
                                    <tr>
                                        <td scope="row">Classifier</td>
                                        <td>
                                            <Tooltip title="(size last hidden layer, number of known classes)">
                                                <div>
                                                    ({(props.pbn_hidden_layers.length > 0)
                                                    ? props.pbn_hidden_layers[props.pbn_hidden_layers.length - 1]
                                                    : '?'}, {format_input_output_size(props.n_known_classes)})
                                                </div>
                                            </Tooltip>
                                        </td>
                                        <td>{(props.pbn_hidden_layers.length > 0 && props.n_known_classes !== null)
                                            ? props.pbn_hidden_layers[props.pbn_hidden_layers.length - 1] * props.n_known_classes + props.n_known_classes
                                            : '?'}</td>
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

export default PBNParameters;