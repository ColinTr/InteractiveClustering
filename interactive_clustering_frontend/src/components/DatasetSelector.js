import Container from "react-bootstrap/Container";
import React from 'react';
import Col from "react-bootstrap/Col";
import Row from "react-bootstrap/Row";

import LineNavigator from "line-navigator";


class DatasetSelector extends React.Component {

    state = {
        selectedFile: null
    };

    onFileChange = event => {
        this.setState({ selectedFile: event.target.files[0] });
    };

    onFileUpload = event => {
        const formData = new FormData();

        formData.append(
            "myFile",
            this.state.selectedFile,
            this.state.selectedFile.name
        );

        console.log(this.state.selectedFile);

        const navigator = new LineNavigator(this.state.selectedFile);
        navigator.readLines(0, 1, function (err, index, lines, isEof, progress) {
            console.log(lines)
        })
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