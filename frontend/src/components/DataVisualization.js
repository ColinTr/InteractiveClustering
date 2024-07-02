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
import Row from "react-bootstrap/Row";
import Col from "react-bootstrap/Col";
import Button from "react-bootstrap/Button";
import { Tooltip } from "@mui/material";
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome'
import { regular } from '@fortawesome/fontawesome-svg-core/import.macro'
import Plot from 'react-plotly.js';


class DataVisualization extends React.Component {
    render() {
        return (
            <Container style={{height:"100%"}}>
                <Col className="d-flex flex-column" style={{height: '100%', paddingLeft: "6px"}}>
                    <Row className="d-flex flex-row" style={{paddingRight: "6px"}}>
                        <div style={{display: "flex"}}>
                            <h5>Data visualization</h5>
                            {/*
                                <Tooltip title="Save image" style={{marginLeft: "auto", marginRight: "6px"}}>
                                    <Button className="btn-secondary" onClick={this.props.onSaveImageButtonClick}>
                                        <div className="d-flex py-1">
                                            <FontAwesomeIcon icon={regular('floppy-disk')}/>
                                        </div>
                                    </Button>
                                </Tooltip>
                            */}
                            <Tooltip title="Clear cached data in server" style={{marginLeft: "auto", marginRight: "6px"}}>
                                <Button className="btn-secondary" onClick={this.props.onClearCacheButtonClick}>
                                    <div className="d-flex py-1">
                                        <FontAwesomeIcon icon={regular('trash-can')}/>
                                    </div>
                                </Button>
                            </Tooltip>
                        </div>
                    </Row>
                    <Row className="d-flex flex-row mt-1" style={{flexGrow:'1', overflowY: "auto", height:"100%"}}>
                        <center>
                            <div style={{display: "flex", alignItems: "center", height: "100%", justifyContent: "center"}}>
                                <Plot data={this.props.image_to_display === null ? [] : this.props.image_to_display.data}
                                      layout={this.props.image_to_display === null ? {} : this.props.image_to_display.layout}
                                      style={{height: "97%", width: "100%", objectFit: "contain"}}
                                      alt="T-SNE of the data"
                                      onClick={(e) => this.props.handlePointClick(e)}
                                      useResizeHandler={true}
                                />
                            </div>
                        </center>
                    </Row>
                    <Row className="d-flex flex-row" style={{paddingLeft: "6px", paddingRight: "6px"}}>
                        <Col className="d-flex flex-column">
                            <Row className="d-flex flex-row">
                                <div style={{display: "flex"}}>
                                    <Button onClick={() => this.props.onRawDataButtonClick()} style={{marginRight: "10px"}}>
                                        View data
                                    </Button>
                                </div>
                            </Row>
                        </Col>
                        <Col className="d-flex flex-column">
                            <Row className="d-flex flex-row" style={{height:"100%"}}>
                                <div className="form-check form-switch justify-content-end" style={{display: "flex", alignItems: "center"}}>
                                    <input className="form-check-input"
                                           type="checkbox"
                                           key="switch_show_unknown_only"
                                           id="switch_show_unknown_only"
                                           onChange={() => this.props.onShowUnknownOnlySwitchChange()}
                                           checked={this.props.show_unknown_only}
                                           style={{marginRight: "10px"}}
                                    />
                                    <label className="form-check-label" htmlFor="switch_show_unknown_only">
                                        Show unknown classes only
                                    </label>
                                </div>
                            </Row>
                        </Col>
                    </Row>
                </Col>
            </Container>
        )
    }
}

export default DataVisualization;