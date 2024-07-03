/*
Software Name : InteractiveClustering
Version: 1.0
SPDX-FileCopyrightText: Copyright (c) 2024 Orange
SPDX-License-Identifier: MIT
This software is distributed under the MIT License,
the text of which is available at https://spdx.org/licenses/MIT.html
or see the "license.txt" file for more details.
*/

import React from 'react';
import Row from "react-bootstrap/Row";
import Col from "react-bootstrap/Col";
import RangeSlider from 'react-bootstrap-range-slider';
import Container from "react-bootstrap/Container";
import Button from 'react-bootstrap/Button';


class AgglomerativeClustering extends React.Component {

    constructor(props) {
        super(props);
        this.state = {
            agglomerative_clustering_value: 1
        };
    }

    render() {
        return (
            <Container>
                <Col className="d-flex flex-column" style={{height: '100%', paddingLeft: "6px", paddingRight: "6px"}}>
                    <Row className="d-flex flex-row pb-1">
                        <h5>Agglomerative clustering</h5>
                    </Row>

                    <Row className="d-flex flex-row">
                        <Col className="col-7 d-flex flex-column"
                             style={{flexGrow:'1'}}>
                            <RangeSlider
                                value={this.state.agglomerative_clustering_value}
                                min={1}
                                max={42}  // To set to the total number of clusters
                                tooltip='off'
                                onChange={changeEvent => this.setState({agglomerative_clustering_value: changeEvent.target.value})}
                                // ToDo control component's value directly with value={...}
                            />
                        </Col>

                        <Col className="col-2 d-flex flex-column justify-content-center align-items-center">
                            {this.state.agglomerative_clustering_value}
                        </Col>
                    </Row>
                    <Row className="d-flex flex-row">
                        <Col className="col-6 d-flex flex-column">
                            <Button style={{width:'120px'}} onClick={() => this.props.onAgglomerativeClusteringRunButtonClick(this.state.agglomerative_clustering_value)}>
                                Run
                            </Button>
                        </Col>
                        <Col className="col-6 d-flex flex-column flex-column align-items-end">
                            <Button variant="success" style={{width:'120px'}} onClick={() => this.props.onAgglomerativeClusteringUpdateRulesButtonClick()}>
                                Update rules
                            </Button>
                        </Col>
                    </Row>
                </Col>
            </Container>
        )
    }
}

export default AgglomerativeClustering;