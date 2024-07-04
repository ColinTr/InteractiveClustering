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
import { faRotate } from "@fortawesome/free-solid-svg-icons";


class DataVisualization extends React.Component {

    // constructor(props) {
    //     super(props);
    //
    //     this.state = {
    //         lastLegendClickTime: 0
    //     }
    // }

    render() {
        return (
            <Container style={{height:"100%"}}>
                <Col className="d-flex flex-column" style={{height: '100%', paddingLeft: "6px"}}>
                    <Row className="d-flex flex-row" style={{paddingRight: "6px"}}>
                        <div style={{display: "flex"}}>
                            <Col className="col-auto d-flex flex-column" style={{justifyContent: "center", flexGrow:'1', textAlign: "left"}}>
                                <h5>Data visualization</h5>
                            </Col>
                            <Col className="col-auto d-flex flex-column" style={{justifyContent: "center"}}>
                                <Tooltip title="Reset all parameters to default" style={{marginLeft: "auto", marginRight: "6px"}}>
                                    <Button className="btn-secondary" onClick={this.props.onResetParametersToDefault}>
                                        <div className="d-flex py-1">
                                            <FontAwesomeIcon icon={faRotate} />
                                        </div>
                                    </Button>
                                </Tooltip>
                            </Col>
                            <Col className="col-auto d-flex flex-column" style={{justifyContent: "center"}}>
                                <Tooltip title="Clear cached data in server" style={{marginLeft: "auto", marginRight: "6px"}}>
                                    <Button className="btn-secondary" onClick={this.props.onClearCacheButtonClick}>
                                        <div className="d-flex py-1">
                                            <FontAwesomeIcon icon={regular('trash-can')}/>
                                        </div>
                                    </Button>
                                </Tooltip>
                            </Col>
                        </div>
                    </Row>
                    <Row className="d-flex flex-row mt-1" style={{flexGrow:'1', overflowY: "auto", height:"100%"}}>
                        <center>
                            <div style={{display: "flex", alignItems: "center", height: "100%", justifyContent: "center"}}>
                                <Plot data={this.props.image_to_display === null ? [] : this.props.image_to_display.data}
                                      layout={this.props.image_to_display === null ? {} : this.props.image_to_display.layout}
                                      style={{height: "97%", width: "100%", objectFit: "contain"}}
                                      alt="T-SNE of the data"
                                      onLegendClick={(e) => this.handleLegendClick(e)}
                                      onClick={(e) => {this.props.handlePointClick(e)}}
                                      useResizeHandler={true}
                                      config={{responsive: true}}
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

    handleLegendClick(event) {
        // const currentTime = new Date().getTime();
        // const timeDiff = currentTime - this.state.lastLegendClickTime;
        // this.setState({lastLegendClickTime: currentTime});
        //
        // if (timeDiff < 300) {
        //     // Double click behavior
        //     console.log('Double click on legend:', event);
        // } else {
        //     // Single click behavior
        //     console.log('Single click on legend:', event);
        // }

        const group_title = event.node.__data__[0].groupTitle

        // If the clicked element has a group_title, it's a legend group title
        if (group_title) {
            const clicked_legend_group_title = group_title.text

            const some_traces_are_visible = this.props.image_to_display.data.some(trace =>
                (clicked_legend_group_title === "Generated clusters" && trace.legendgroup === "clusters" && trace.visible === true)
                || (clicked_legend_group_title === "Unknown data" && trace.legendgroup === "unknown" && trace.visible === true)
                || (clicked_legend_group_title === "Known classes" && trace.legendgroup === "classes" && trace.visible === true)
            )

            const updatedData = this.props.image_to_display.data.map(trace => {
                if ((clicked_legend_group_title === "Generated clusters" && trace.legendgroup === "clusters")
                    || (clicked_legend_group_title === "Unknown data" && trace.legendgroup === "unknown")
                    || (clicked_legend_group_title === "Known classes" && trace.legendgroup === "classes")) {
                    // If some traces in this group are visible, set them all to 'legendonly'
                    // Otherwise, set all their visibility to true
                    return { ...trace, visible: some_traces_are_visible ? 'legendonly' : true }
                } else {
                    return trace
                }
            })

            // Update the plot with the new data
            this.props.updateImageToDisplayData(updatedData)
        }
        // Otherwise, it's an individual element of the legend
        else {
            const traceIndex = event.curveNumber;  // This is the index of the clicked element in the legend
            const updatedData = this.props.image_to_display.data.map((trace, i) => {
                if (i === traceIndex) {
                    return {...trace, visible: trace.visible === true ? 'legendonly' : true}
                } else {
                    return trace
                }
            })

            // Update the plot with the new data
            this.props.updateImageToDisplayData(updatedData)
        }

        return false; // Prevent default legend item toggle behavior
    }
}

export default DataVisualization;
