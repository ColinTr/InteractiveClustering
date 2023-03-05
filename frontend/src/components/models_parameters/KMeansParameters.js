import React from "react";
import Container from "react-bootstrap/Container";
import Col from "react-bootstrap/Col";
import Row from "react-bootstrap/Row";
import {Tooltip} from "@mui/material";
import {AiOutlineQuestionCircle} from "react-icons/ai";


const KMeansParameters = (props) => {
    console.log(props.model_params_kmeans_train_on_unknown_classes_only)
    return (
        <Container>
            <Row className="d-flex flex-row" style={{marginBottom: "10px"}}>
                <Col className="col-8 d-flex flex-column">
                    <Tooltip title="The number of classes to predict">
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
                           onChange={props.on_kmeans_n_clusters_change}
                           defaultValue={props.n_clusters_value}
                    />
                </Col>
            </Row>

            <Row className="d-flex flex-row" style={{marginBottom: "10px"}}>
                <Col className="col-10 d-flex flex-column">
                    <Tooltip title="Fit the k-means model only on the unknown classes, or on all the data (known + unknown)">
                        <label className="form-check-label" htmlFor="switch_show_model_prediction">
                            Train on unknown classes only <AiOutlineQuestionCircle style={{marginLeft: "5px"}}/>
                        </label>
                    </Tooltip>
                </Col>
                <Col className="col-2 d-flex flex-column">
                    <div className="form-check form-switch justify-content-center" style={{display: "flex", alignItems: "center"}}>
                        <input className="form-check-input"
                               type="checkbox"
                               key="switch_show_model_prediction"
                               id="switch_show_model_prediction"
                               onChange={props.onKMeansTrainOnUknownClassesOnlySwitchChange}
                               checked={props.model_params_kmeans_train_on_unknown_classes_only}
                               style={{marginRight: "10px"}}
                        />
                    </div>
                </Col>
            </Row>
        </Container>
    )

}

export default KMeansParameters;