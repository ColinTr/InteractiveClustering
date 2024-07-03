/*
Software Name : InteractiveClustering
Version: 1.0
SPDX-FileCopyrightText: Copyright (c) 2024 Orange
SPDX-License-Identifier: MIT
This software is distributed under the MIT License,
the text of which is available at https://spdx.org/licenses/MIT.html
or see the "license.txt" file for more details.
*/

import Container from "react-bootstrap/Container";
import React from 'react';
import Col from "react-bootstrap/Col";
import Row from "react-bootstrap/Row";
import {FontAwesomeIcon} from "@fortawesome/react-fontawesome";
import {regular} from "@fortawesome/fontawesome-svg-core/import.macro";
import Tooltip from "@mui/material/Tooltip";

class DatasetSelector extends React.Component {
    render() {
        return (
            <Container>
                <Col className="d-flex flex-column" style={{height: "100%", paddingLeft: "6px", paddingRight: "6px"}}>
                    <Row className="d-flex flex-row">
                        <h5>Dataset selection</h5>
                    </Row>
                    <Row className="d-flex flex-row">
                        <Col className="col-12 d-flex flex-column justify-content-center">
                            <input type="file" onChange={(event) => this.props.onFileChange(event)} style={{backgroundColor: "white", color: "black"}} id="my_input_file_form"/>
                        </Col>
                    </Row>
                    <Row className="d-flex flex-row pt-2">
                        <Col className="col-6 d-flex flex-column justify-content-center">
                            <Tooltip title={<div>Character used as field separator in the file.<br/>For tabulation, use \t.<br/>Regular expressions like \s+ are also supported.</div>}>
                                <div style={{display: "flex", alignItems: "center"}}>
                                    Field separator <FontAwesomeIcon icon={regular('circle-question')} style={{marginLeft: "5px"}}/>
                                </div>
                            </Tooltip>
                        </Col>
                        <Col className="col-6 d-flex flex-column justify-content-center">
                            <input type="text"
                                   placeholder="field separator"
                                   onChange={this.props.onDatasetSeparatorChange}
                                   value={this.props.field_separator}
                            />
                        </Col>
                    </Row>
                    <Row className="d-flex flex-row pt-2">
                        <Col className="col-6 d-flex flex-column" style={{textAlign: "right"}}>
                            <button type="button" className="btn btn-primary" onClick={this.props.onFileUpload} style={{width:'100px'}}>Load</button>
                        </Col>
                        <Col className="col-6 d-flex flex-column align-items-end" style={{textAlign: "right"}}>
                            <button type="button" className="btn btn-danger" onClick={this.props.onDatasetUnload} style={{width:'100px'}}>Unload</button>
                        </Col>
                    </Row>
                </Col>
            </Container>
        );
    }
}

export default DatasetSelector;