/*
Software Name : InteractiveClustering
Version: 1.0
SPDX-FileCopyrightText: Copyright (c) 2024 Orange
SPDX-License-Identifier: MIT
This software is distributed under the MIT License,
the text of which is available at https://spdx.org/licenses/MIT.html
or see the "license.txt" file for more details.
*/

import React, { forwardRef } from "react"
import { SnackbarContent, closeSnackbar} from "notistack"
import Container from "react-bootstrap/Container";
import Row from "react-bootstrap/Row";
import Col from "react-bootstrap/Col";
import Swal from "sweetalert2";
import FireSwalError from "./FireSwalError";


const ModelTrainingSnackbar = forwardRef((props, ref) => {
    function onCancelTrainingButtonClick(props) {
        clearInterval(props.refreshIntervalId)

        const cancelThreadRequestOptions = {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({'thread_id': props.thread_id})
        }
        fetch('/cancelTrainingThread', cancelThreadRequestOptions).then(cancelServerPromise => {
            if (cancelServerPromise.status === 500) {
                FireSwalError('Status 500 - Server error', 'Please make sure that the server is running')
            }
            if (cancelServerPromise.status === 422) {
                cancelServerPromise.json().then(error => {
                    FireSwalError('Status 422 - Server error', error['error_message'])
                })
            }
            if (cancelServerPromise.status === 200) {
                Swal.mixin({
                    toast: true,
                    position: 'top-end',
                    showConfirmButton: false,
                    timer: 3000,
                    timerProgressBar: true,
                    didOpen: (toast) => {
                        toast.addEventListener('mouseenter', Swal.stopTimer)
                        toast.addEventListener('mouseleave', Swal.resumeTimer)
                    }
                }).fire({
                    icon: 'success',
                    title: "Stopped thread " + props.thread_id
                })
            }

            // In any case, we need to close this snackbar
            closeSnackbar(props.id)
        })
    }

    return (
        <SnackbarContent ref={ref}>
            <Container style={{backgroundColor: "white", color: "black", borderRadius: "5px", paddingLeft: "10px", paddingRight: "10px", paddingBottom: "10px", paddingTop: "5px"}}>
                <Col>
                    <Row>
                        <div style={{textAlign: "left"}}>
                            Started training model...
                        </div>
                    </Row>
                    <Row style={{marginTop: "10px", marginBottom: "5px"}}>
                        <div>
                            <div className="progress">
                                <div className="progress-bar progress-bar-striped progress-bar-animated" role="progressbar" aria-valuenow="" aria-valuemin="0" aria-valuemax="100" id={"pb_thread_" + props.thread_id}/>
                            </div>
                        </div>
                    </Row>
                    <Row>
                        <div className="text-start" id={"time_estimation_thread_" + props.thread_id}>Remaining time: ...</div>
                    </Row>
                    <Row>
                        <Col className="col-8 d-flex flex-column" style={{width:"100%"}}>
                            <div className="form-check form-switch" style={{display: "flex", alignItems: "center", width:"100%"}}>
                                <input className="form-check-input"
                                       type="checkbox"
                                       key="switch_show_unknown_only"
                                       id="switch_show_unknown_only"
                                       onChange={() => props.onViewInEncoderSwitchChange(props.thread_id)}
                                       style={{marginRight: "10px"}}
                                />
                                <label className="form-check-label" htmlFor="switch_show_unknown_only">
                                    View in encoder
                                </label>
                            </div>
                        </Col>
                    </Row>
                    <Row>
                        <Col className="col-4 d-flex flex-column align-items-start">
                            <button type="button" className="btn btn-danger" onClick={() => onCancelTrainingButtonClick(props)}>
                                Cancel
                            </button>
                        </Col>
                        <Col className="col-8 d-flex flex-column align-items-end">
                            <button type="button" className="btn btn-success" onClick={() => props.onSeeResultsButtonClick(props.thread_id)}>
                                See results
                            </button>
                        </Col>
                    </Row>
                </Col>
            </Container>
        </SnackbarContent>
    )
})

export default ModelTrainingSnackbar;