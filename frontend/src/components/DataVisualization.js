import React from "react";
import Container from "react-bootstrap/Container";
import Row from "react-bootstrap/Row";
import Col from "react-bootstrap/Col";
import Button from "react-bootstrap/Button";
import {Tooltip} from "@mui/material";
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome'
import { regular } from '@fortawesome/fontawesome-svg-core/import.macro'
import Swal from "sweetalert2";
import fireSwalError from "./swal_functions";


class DataVisualization extends React.Component {
    img_display = () => {
        if(this.props.image_to_display == null){
            return (
                <div style={{display: "flex", alignItems: "center", height:"100%", justifyContent: "center"}}>
                    <i>Load a dataset to visualize</i>
                </div>
            )
        } else {
            return (
                <img src={this.props.image_to_display}
                     alt="T-SNE of the data"
                     style={{height: "100%", width:"88%", objectFit: "contain"}}
                />
            )
        }
    }

    onClearCacheButtonClick = () => {
        Swal.fire({
            title: 'Are you sure?',
            text: "Clearing the server\'s temporary files might increase the processing time of the next requests.",
            showDenyButton: true,
            confirmButtonText: 'Clear',
            denyButtonText: `Don't clear`,
        }).then((result) => {
            if (result.isConfirmed) {
                const requestOptions = {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' }
                }
                fetch('/clearServerCache', requestOptions)
                    .then(serverPromise => {
                        if (serverPromise.status === 500) {
                            fireSwalError('Status 500 - Server error', 'Please make sure that the server is running')
                        }
                        if (serverPromise.status === 422) {
                            serverPromise.json().then(error => {
                                fireSwalError('Status 422 - Server error', error['error_message'])
                            })
                        }
                        if (serverPromise.status === 200) {
                            Swal.fire('Done!', '', 'success')
                        }
                    })
            }
        })
    }

    render() {
        return (
            <Container style={{height:"100%"}}>
                <Col className="d-flex flex-column" style={{height: '100%', paddingLeft: "6px"}}>
                    <Row className="d-flex flex-row" style={{paddingRight: "6px"}}>
                        <div style={{display: "flex"}}>
                            <h5>Data visualization</h5>
                                <Tooltip title="Save image" style={{marginLeft: "auto", marginRight: "6px"}}>
                                    <Button className="btn-secondary" onClick={this.props.onSaveImageButtonClick}>
                                        <div className="d-flex py-1">
                                            <FontAwesomeIcon icon={regular('floppy-disk')}/>
                                        </div>
                                    </Button>
                                </Tooltip>
                                <Tooltip title="Clear cached data in server">
                                    <Button className="btn-secondary" onClick={this.onClearCacheButtonClick}>
                                        <div className="d-flex py-1">
                                            <FontAwesomeIcon icon={regular('trash-can')}/>
                                        </div>
                                    </Button>
                                </Tooltip>
                        </div>
                    </Row>
                    <Row className="d-flex flex-row mt-1" style={{flexGrow:'1', overflowY: "auto", height:"100%"}}>
                        <center>
                            {this.img_display()}
                        </center>
                    </Row>
                    <Row className="d-flex flex-row" style={{paddingLeft: "6px", paddingRight: "6px"}}>
                        <Col className="d-flex flex-column">
                            <Row className="d-flex flex-row " style={{paddingLeft: "12px"}}>
                                Get T-SNE of the...
                            </Row>
                            <Row className="d-flex flex-row">
                                <div style={{display: "flex"}}>
                                    <Button onClick={() => this.props.onRawDataButtonClick()} style={{marginRight: "10px"}}>
                                        Raw data
                                    </Button>
                                    <Button onClick={() => this.props.onProjectionButtonClick()}>
                                        Projection
                                    </Button>
                                </div>
                            </Row>
                        </Col>
                        <Col className="d-flex flex-column">
                            <Row className="d-flex flex-row" style={{height:"100%"}}>
                                <div className="form-check form-switch" style={{display: "flex", alignItems: "center"}}>
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