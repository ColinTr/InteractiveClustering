import Container from "react-bootstrap/Container";
import React from 'react';
import Col from "react-bootstrap/Col";
import Row from "react-bootstrap/Row";


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

        // Request made to the backend api
        // Send formData object
        // axios.post("api/uploadfile", formData);
    };

    render() {
        return (
            <Container>
                <h5>Dataset selection</h5>
                <Row>
                    <Col className="col-lg-10 col-md-12 mb-2">
                        <div className="mr-3" style={{backgroundColor: "white", color: "black"}}>
                            <input type="file" onChange={this.onFileChange}/>
                        </div>
                    </Col>
                    <Col className="col-lg-2 col-md-12" style={{textAlign: "right"}}>
                        <button onClick={this.onFileUpload}>Load</button>
                    </Col>
                </Row>
            </Container>
        );
    }
}

export default DatasetSelector;