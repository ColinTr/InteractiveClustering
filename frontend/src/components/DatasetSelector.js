import Container from "react-bootstrap/Container";
import React from 'react';
import Col from "react-bootstrap/Col";
import Row from "react-bootstrap/Row";
import fireSwalError from "./swal_functions";
import {FontAwesomeIcon} from "@fortawesome/react-fontawesome";
import {regular} from "@fortawesome/fontawesome-svg-core/import.macro";
import Tooltip from "@mui/material/Tooltip";

class DatasetSelector extends React.Component {

    constructor(props) {
        super(props);
        this.state = {
            selectedFile: "",
            dataset_separator: ',',
        };
    }

    onFileChange = event => {
        this.setState({ selectedFile: event.target.files[0].name })
    }

    onDatasetUnload = () => {
        this.setState({selectedFile: ""})
        document.getElementById("my_input_file_form").value = "";
        this.props.unloadDatasetHandler()
    }

    onDatasetSeparatorChange = selected_sep => {
        this.setState({dataset_separator: selected_sep.target.value})
    }

    onFileUpload = () => {
        if (this.state.selectedFile === ""){
            fireSwalError('Please select a file to load')
            return
        }

        if (this.state.dataset_separator === ""){
            fireSwalError('Please specify a field separator')
            return
        }

        const dataset_name = this.state.selectedFile.replace(/\.[^/.]+$/, "")
        const requestOptions = {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                'selected_file_path': this.state.selectedFile,
                'field_separator': this.state.dataset_separator,
                'dataset_name': dataset_name})
        }
        fetch('/getFileHeader', requestOptions)   // Don't need to specify the full localhost:5000/... as the proxy is set in package.json
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
                    serverPromise.json().then(response => {
                        // The features we just received from the server are sent to the FullPage.js component
                        this.props.onNewFeaturesLoaded(response['file_header'])
                        this.props.setDatasetNameHandler(dataset_name)
                    })
                }
            })
    };

    render() {
        return (
            <Container>
                <Col className="d-flex flex-column" style={{height: "100%", paddingLeft: "6px", paddingRight: "6px"}}>
                    <Row className="d-flex flex-row">
                        <h5>Dataset selection</h5>
                    </Row>
                    <Row className="d-flex flex-row">
                        <Col className="col-12 d-flex flex-column justify-content-center">
                            <input type="file" onChange={this.onFileChange} style={{backgroundColor: "white", color: "black"}} id="my_input_file_form"/>
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
                                   onChange={this.onDatasetSeparatorChange}
                                   defaultValue={this.state.dataset_separator}
                            />
                        </Col>
                    </Row>
                    <Row className="d-flex flex-row pt-2">
                        <Col className="col-6 d-flex flex-column" style={{textAlign: "right"}}>
                            <button type="button" className="btn btn-primary" onClick={this.onFileUpload} style={{width:'100px'}}>Load</button>
                        </Col>
                        <Col className="col-6 d-flex flex-column align-items-end" style={{textAlign: "right"}}>
                            <button type="button" className="btn btn-danger" onClick={this.onDatasetUnload} style={{width:'100px'}}>Unload</button>
                        </Col>
                    </Row>
                </Col>
            </Container>
        );
    }
}

export default DatasetSelector;