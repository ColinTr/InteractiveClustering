import React from "react";
import Container from "react-bootstrap/Container";
import Col from "react-bootstrap/Col";
import Row from "react-bootstrap/Row";
import { Tooltip } from "@mui/material";
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome'
import { regular } from '@fortawesome/fontawesome-svg-core/import.macro'


const KMeansParameters = (props) => {
    return (
        <Container>
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
                           placeholder="Number of clusters"
                           step={1}
                           onChange={props.on_kmeans_n_clusters_change}
                           defaultValue={props.k_means_n_clusters}
                    />
                </Col>
            </Row>
        </Container>
    )

}

export default KMeansParameters;