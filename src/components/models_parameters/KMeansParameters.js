import React from "react";
import Container from "react-bootstrap/Container";
import Col from "react-bootstrap/Col";
import Row from "react-bootstrap/Row";


const KMeansParameters = (props) => {
    return (
        <Container>
            <Row>
                <Col className="col-8 d-flex flex-column">
                    <p>Number of clusters</p>
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