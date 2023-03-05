import React from "react";
import Container from "react-bootstrap/Container";
import Row from "react-bootstrap/Row";
import Col from "react-bootstrap/Col";
import Form from "react-bootstrap/Form";
import {Tooltip} from "@mui/material";
import {AiOutlineQuestionCircle} from "react-icons/ai";


const TabularNCDParameters = (props) => {
    return (
        <Container>
            <Row className="d-flex flex-row" style={{marginBottom: "10px"}}>
                <Col className="col-8 d-flex flex-column">
                    <Tooltip title="The number of unknown classes to predict">
                        <div style={{display: "flex", alignItems: "center"}}>
                            Number of clusters <AiOutlineQuestionCircle style={{marginLeft: "5px"}}/>
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

            <Row className="d-flex flex-row" style={{marginBottom: "10px"}}>
                <Col className="col-8 d-flex flex-column">
                    <Tooltip title="ToDo tooltip">
                        <div style={{display: "flex", alignItems: "center"}}>
                            Cosine Top k (in %) <AiOutlineQuestionCircle style={{marginLeft: "5px"}}/>
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

            <Row className="d-flex flex-row" style={{marginBottom: "10px"}}>
                <Col className="col-8 d-flex flex-column">
                    <Tooltip title="ToDo tooltip">
                        <div style={{display: "flex", alignItems: "center"}}>
                            w1 <AiOutlineQuestionCircle style={{marginLeft: "5px"}}/>
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
                            w2 <AiOutlineQuestionCircle style={{marginLeft: "5px"}}/>
                        </div>
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
                            Classif. lr <AiOutlineQuestionCircle style={{marginLeft: "5px"}}/>
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
                            Clust. lr <AiOutlineQuestionCircle style={{marginLeft: "5px"}}/>
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
                            k neighbors <AiOutlineQuestionCircle style={{marginLeft: "5px"}}/>
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
                            Dropout <AiOutlineQuestionCircle style={{marginLeft: "5px"}}/>
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

            <Row className="d-flex flex-row">
                <Col className="col-7 d-flex flex-column">
                    <Tooltip title="ToDo tooltip">
                        <div style={{display: "flex", alignItems: "center"}}>
                            Activation function <AiOutlineQuestionCircle style={{marginLeft: "5px"}}/>
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
                </Col>s
            </Row>
        </Container>
    )
}

export default TabularNCDParameters;