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

const TabularNCDParameters = (props) => {
    return (
        <Container>
            <Row className="d-flex flex-row" style={{marginBottom: "10px"}}>
                <Col className="col-12 d-flex flex-column">
                    <NoMaxWidthTooltip title={<img src="/TabularNCD_architecture.png" alt="Model architecture" width="700"/>} placement="bottom-end">
                        <div style={{display: "flex", alignItems: "center"}}>
                            <u>Model architecture help</u>
                        </div>
                    </NoMaxWidthTooltip >
                </Col>
            </Row>

            <Row className="d-flex flex-row" style={{marginBottom: "10px"}}>
                <Col className="col-8 d-flex flex-column">
                    <Tooltip title="The number of unknown classes to predict">
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
                           onChange={props.on_tabncd_n_clusters_change}
                           defaultValue={props.tabncd_n_clusters_value}
                    />
                </Col>
            </Row>

            {/*
            <Row className="d-flex flex-row" style={{marginBottom: "10px"}}>
                <Col className="col-8 d-flex flex-column">
                    <Tooltip title="ToDo tooltip">
                        <div style={{display: "flex", alignItems: "center"}}>
                            Cosine Top k (in %) <FontAwesomeIcon icon={regular('circle-question')} style={{marginLeft: "5px"}}/>
                        </div>
                    </Tooltip>
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
            */}

            <Row className="d-flex flex-row" style={{marginBottom: "10px"}}>
                <Col className="col-8 d-flex flex-column">
                    <Tooltip title="ToDo tooltip">
                        <div style={{display: "flex", alignItems: "center"}}>
                            w1 <FontAwesomeIcon icon={regular('circle-question')} style={{marginLeft: "5px"}}/>
                        </div>
                    </Tooltip>
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

            <Row className="d-flex flex-row" style={{marginBottom: "10px"}}>
                <Col className="col-8 d-flex flex-column">
                    <Tooltip title="ToDo tooltip">
                        <div style={{display: "flex", alignItems: "center"}}>
                            w2 <FontAwesomeIcon icon={regular('circle-question')} style={{marginLeft: "5px"}}/></div>
                    </Tooltip>
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

            <Row className="d-flex flex-row" style={{marginBottom: "10px"}}>
                <Col className="col-8 d-flex flex-column">
                    <Tooltip title="ToDo tooltip">
                        <div style={{display: "flex", alignItems: "center"}}>
                            Classif. lr <FontAwesomeIcon icon={regular('circle-question')} style={{marginLeft: "5px"}}/>
                        </div>
                    </Tooltip>
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

            <Row className="d-flex flex-row" style={{marginBottom: "10px"}}>
                <Col className="col-8 d-flex flex-column">
                    <Tooltip title="ToDo tooltip">
                        <div style={{display: "flex", alignItems: "center"}}>
                            Clust. lr <FontAwesomeIcon icon={regular('circle-question')} style={{marginLeft: "5px"}}/>
                        </div>
                    </Tooltip>
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

            <Row className="d-flex flex-row" style={{marginBottom: "10px"}}>
                <Col className="col-8 d-flex flex-column">
                    <Tooltip title="ToDo tooltip">
                        <div style={{display: "flex", alignItems: "center"}}>
                            k neighbors <FontAwesomeIcon icon={regular('circle-question')} style={{marginLeft: "5px"}}/>
                        </div>
                    </Tooltip>
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
                           onChange={props.on_tabncd_dropout_change}
                           defaultValue={props.tabncd_dropout_value}
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
                        onChange={props.on_tabncd_activation_fct_change}
                        style={{paddingTop: 0, paddingLeft: "3px", paddingBottom: 0}}
                    >
                        <option value="sigmoid">Sigmoid</option>
                        <option value="relu">ReLu</option>
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
                                    {props.tabncd_hidden_layers.map((layer_size, layer_index) => (
                                        <tr key={"hidden_layer_row_" + layer_index}>
                                            <td scope="row">Dense {layer_index}</td>
                                            <td>({(layer_index === 0)
                                                ? format_input_output_size(props.n_features_used)
                                                : props.tabncd_hidden_layers[layer_index - 1]}, {layer_size})</td>
                                            <td>{(layer_index === 0)
                                                ? (props.n_features_used === null)
                                                    ? '?'
                                                    : props.n_features_used * layer_size + layer_size
                                                : props.tabncd_hidden_layers[layer_index - 1] * layer_size + layer_size}</td>
                                            <td>
                                                <Button className="btn-secondary"
                                                        style={{borderRadius: "50px", padding: "3px", paddingRight: "2px", paddingBottom: "1px", paddingTop: "1px"}}
                                                        onClick={() => props.onTabncdRemoveLayerButtonClick(layer_index)}
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
                                                <input id="tabncdLayerSizeInput"
                                                       type="number"
                                                       min={0}
                                                       placeholder="size"
                                                       step={1}
                                                       style={{marginLeft: "5px", marginRight: "5px", maxWidth: "60px"}}
                                                />
                                                <Button className="btn-secondary"
                                                        style={{borderRadius: "50px", padding: "3px", paddingRight: "2px", paddingBottom: "1px", paddingTop: "1px"}}
                                                        onClick={props.onTabncdAddLayerButtonClick}
                                                >
                                                    <div className="d-flex align-items-center">
                                                        <FontAwesomeIcon icon={solid('plus')}/>
                                                    </div>
                                                </Button>
                                            </center>
                                        </td>
                                    </tr>
                                    </tbody>
                                </table>
                            </div>
                        </Row>
                    </Col>
                </div>
            </Row>

            <Row className="d-flex flex-row" style={{marginBottom: "10px"}}>
                <div>
                    <Col className="d-flex flex-column">
                        <Row>
                            <div>
                                Clustering layer:
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
                                            <Tooltip title="(Output size of the encoder)">
                                                <div>
                                                    ({(props.tabncd_hidden_layers.length > 0)
                                                    ? props.tabncd_hidden_layers[props.tabncd_hidden_layers.length - 1]
                                                    : '?'})
                                                </div>
                                            </Tooltip>
                                        </td>
                                        <td></td>
                                        <td></td>
                                    </tr>
                                    <tr>
                                        <td scope="row">Output</td>
                                        <td>
                                            <Tooltip title="(size last hidden layer, number of known classes)">
                                                <div>
                                                    ({(props.tabncd_hidden_layers.length > 0)
                                                    ? props.tabncd_hidden_layers[props.tabncd_hidden_layers.length - 1]
                                                    : '?'}, {format_input_output_size(props.tabncd_n_clusters_value)})
                                                </div>
                                            </Tooltip>
                                        </td>
                                        <td>{(props.tabncd_hidden_layers.length > 0 && props.tabncd_n_clusters_value !== null)
                                                ? props.tabncd_hidden_layers[props.tabncd_hidden_layers.length - 1] * props.tabncd_n_clusters_value + props.tabncd_n_clusters_value
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

            <Row className="d-flex flex-row" style={{marginBottom: "10px"}}>
                <div>
                    <Col className="d-flex flex-column">
                        <Row>
                            <div>
                                Classification layer:
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
                                            <Tooltip title="(Output size of the encoder)">
                                                <div>
                                                    ({(props.tabncd_hidden_layers.length > 0)
                                                    ? props.tabncd_hidden_layers[props.tabncd_hidden_layers.length - 1]
                                                    : '?'})
                                                </div>
                                            </Tooltip>
                                        </td>
                                        <td></td>
                                        <td></td>
                                    </tr>
                                    <tr>
                                        <td scope="row">Output</td>
                                        <td>
                                            <Tooltip title="(size last hidden layer, number of known classes)">
                                                <div>
                                                    ({(props.tabncd_hidden_layers.length > 0)
                                                    ? props.tabncd_hidden_layers[props.tabncd_hidden_layers.length - 1]
                                                    : '?'}, {format_input_output_size(props.n_known_classes)})
                                                </div>
                                            </Tooltip>
                                        </td>
                                        <td>{(props.tabncd_hidden_layers.length > 0 && props.n_known_classes !== null)
                                            ? props.tabncd_hidden_layers[props.tabncd_hidden_layers.length - 1] * props.n_known_classes + props.n_known_classes
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

export default TabularNCDParameters;