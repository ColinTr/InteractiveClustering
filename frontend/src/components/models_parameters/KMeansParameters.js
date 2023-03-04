import React from "react";
import Container from "react-bootstrap/Container";
import Col from "react-bootstrap/Col";
import Row from "react-bootstrap/Row";
import {Tooltip} from "@mui/material";
import {AiOutlineQuestionCircle} from "react-icons/ai";


const KMeansParameters = (props) => {
    return (
        <Container>
            <Row>
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
                           onChange={props.on_kmeans_n_clusters_change}
                           defaultValue={props.n_clusters_value}
                    />
                </Col>
            </Row>
        </Container>
    )

}

export default KMeansParameters;