import Container from "react-bootstrap/Container";
import React from 'react';
import Col from "react-bootstrap/Col";
import Row from "react-bootstrap/Row";
import Swal from 'sweetalert2'

function fireSwalError(title, text=null){
    Swal.mixin({
        toast: true,
        position: 'top-end',
        showConfirmButton: false,
        timer: 5000,
        timerProgressBar: true,
        didOpen: (toast) => {
            toast.addEventListener('mouseenter', Swal.stopTimer)
            toast.addEventListener('mouseleave', Swal.resumeTimer)
        }
    }).fire({
        icon: 'error',
        title: title,
        text: text
    })
}


class DatasetSelector extends React.Component {

    state = {
        selectedFile: null
    }

    onFileChange = event => {
        this.setState({ selectedFile: event.target.files[0].name })
    }

    onFileUpload = () => {
        if (this.state.selectedFile == null){
            fireSwalError('Please select a file to load')
        } else {
            const requestOptions = {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({'selected_file_path': this.state.selectedFile})
            }
            fetch('/getFileHeader', requestOptions)   // Don't need to specify the full localhost:5000/... as the proxy is set in package.json
                .then(serverPromise => {
                    if (serverPromise.status === 500) {
                        fireSwalError('Status 500 - Internal server error', 'Please make sure that the server is running')
                    }
                    if (serverPromise.status === 200) {
                        serverPromise.json().then(response => {
                            console.log(response['file_header'])
                        })
                    }
                })
        }
    };

    render() {
        return (
            <Container>
                <Col className="d-flex flex-column" style={{height: "100%"}}>
                    <Row className="d-flex flex-row">
                        <h5>Dataset selection</h5>
                    </Row>
                    <Row className="d-flex flex-row">
                        <Col className="col-10 d-flex flex-column justify-content-center">
                            <input type="file" onChange={this.onFileChange} style={{backgroundColor: "white", color: "black"}}/>
                        </Col>
                        <Col className="col-2 d-flex flex-column" style={{textAlign: "right"}}>
                            <button type="button" className="btn btn-primary" onClick={this.onFileUpload}>Load</button>
                        </Col>
                    </Row>
                </Col>
            </Container>
        );
    }
}

export default DatasetSelector;