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
import Col from "react-bootstrap/Col";
import Row from "react-bootstrap/Row";
import { Tooltip } from "@mui/material";
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome'
import { regular } from '@fortawesome/fontawesome-svg-core/import.macro'
import Form from "react-bootstrap/Form";


const SpectralClusteringParameters = (props) => {
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
                           placeholder="n clusters"
                           step={1}
                           onChange={props.on_spectral_clustering_n_clusters_change}
                           value={props.spectral_clustering_n_clusters}
                    />
                </Col>
            </Row>

            <Row className="d-flex flex-row">
                <Col className="col-7 d-flex flex-column">
                    <Tooltip title="How to construct the affinity matrix. ‘Nearest neighbors’ constructs the affinity matrix by computing a graph of nearest neighbors. ‘RBF’ constructs the affinity matrix using a radial basis function (RBF) kernel.">
                        <div style={{display: "flex", alignItems: "center"}}>
                            Affinity <FontAwesomeIcon icon={regular('circle-question')} style={{marginLeft: "5px"}}/>
                        </div>
                    </Tooltip>
                </Col>
                <Col className="col-5 d-flex flex-column">
                    <Form.Select
                        aria-label="Affinity"
                        onChange={props.on_spectral_clustering_affinity_change}
                        value={props.spectral_clustering_affinity}
                        style={{paddingTop: 0, paddingLeft: "3px", paddingBottom: 0}}
                    >
                        <option value="rbf">RBF</option>
                        <option value="nearest_neighbors">Nearest neighbors</option>
                    </Form.Select>
                </Col>
            </Row>
        </Container>
    )

}

export default SpectralClusteringParameters;