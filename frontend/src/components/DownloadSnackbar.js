import React, { forwardRef } from "react"
import { SnackbarContent, closeSnackbar} from "notistack"
import Container from "react-bootstrap/Container";
import Row from "react-bootstrap/Row";
import Col from "react-bootstrap/Col";
import Swal from "sweetalert2";
import fireSwalError from "./swal_functions";


const DownloadSnackbar = forwardRef((props, ref) => {
    function onCancelTrainingButtonClick() {
        clearInterval(props.refreshIntervalId)

        const cancelThreadRequestOptions = {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({'thread_id': props.thread_id})
        }
        fetch('/cancelTrainingThread', cancelThreadRequestOptions).then(cancelServerPromise => {
            if (cancelServerPromise.status === 500) {
                fireSwalError('Status 500 - Server error', 'Please make sure that the server is running')
            }
            if (cancelServerPromise.status === 422) {
                cancelServerPromise.json().then(error => {
                    fireSwalError('Status 422 - Server error', error['error_message'])
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
                    title: 'Training stopped'
                })

                // Close this snackbar
                closeSnackbar(props.id)
            }
        })
    }

    function onDoneButtonClick() {
        // ToDo show the result of the training here...

        // closeSnackbar(props.id)
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
                    <Row style={{marginTop: "10px", marginBottom: "10px"}}>
                        <div>
                            <div className="progress">
                                <div className="progress-bar progress-bar-striped progress-bar-animated" role="progressbar" aria-valuenow="" aria-valuemin="0" aria-valuemax="100" id={"pb_thread_"+props.thread_id}/>
                            </div>
                        </div>
                    </Row>
                    <Row>
                        <Col className="col-6 d-flex flex-column align-items-start">
                            <button type="button" className="btn btn-danger" onClick={onCancelTrainingButtonClick}>
                                Cancel
                            </button>
                        </Col>
                        <Col className="col-6 d-flex flex-column align-items-end">
                            <button type="button" className="btn btn-success" onClick={onDoneButtonClick}>
                                See results
                            </button>
                        </Col>
                    </Row>
                </Col>
            </Container>
        </SnackbarContent>
    )
})

export default DownloadSnackbar