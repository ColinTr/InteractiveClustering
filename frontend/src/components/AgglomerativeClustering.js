import React from 'react';
import Row from "react-bootstrap/Row";
import Col from "react-bootstrap/Col";
import RangeSlider from 'react-bootstrap-range-slider';
import Container from "react-bootstrap/Container";
import Button from 'react-bootstrap/Button';
import fireSwalError from "./swal_functions";


class AgglomerativeClustering extends React.Component {

    constructor(props) {
        super(props);
        this.state = {
            agglomerative_clustering_value: 0
        };
    }

    onRunButtonClick = event => {
        console.log("ToDo: run agglomerative clustering with " + this.state.agglomerative_clustering_value + " clusters fusion")
        fireSwalError("Not implemented yet!")
    };

    render() {
        return (
            <Container>
                <Col className="d-flex flex-column" style={{height: '100%', paddingLeft: "6px", paddingRight: "6px"}}>
                    <Row className="d-flex flex-row pb-1">
                        <h5>Agglomerative clust. granularity</h5>
                    </Row>

                    <Row className="d-flex flex-row">
                        <Col className="col-7 d-flex flex-column"
                             style={{flexGrow:'1'}}>
                            <RangeSlider
                                value={this.state.agglomerative_clustering_value}
                                min={0}
                                max={42}  // To set to the total number of clusters
                                tooltip='off'
                                onChange={changeEvent => this.setState({agglomerative_clustering_value: changeEvent.target.value})}
                            />
                        </Col>

                        <Col className="col-2 d-flex flex-column justify-content-center align-items-center">
                            {this.state.agglomerative_clustering_value}
                        </Col>

                        <Col className="col-3 d-flex flex-column justify-content-center">
                            <Button style={{paddingLeft:0, paddingRight:0}} onClick={() => this.onRunButtonClick()}>
                                Run
                            </Button>
                        </Col>
                    </Row>
                </Col>
            </Container>
        )
    }
}

export default AgglomerativeClustering;