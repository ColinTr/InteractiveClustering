import React from "react";
import Container from "react-bootstrap/Container";
import Row from "react-bootstrap/Row";
import Col from "react-bootstrap/Col";
import Button from "react-bootstrap/Button";


class DataVisualization extends React.Component {
    img_display = () => {
        if(this.props.image_to_display == null){
            return (
                <i>Load a dataset to visualize</i>
            )
        } else {
            return (
                <img src={this.props.image_to_display}
                     alt="T-SNE of the data"
                     style={{height: "100%", width:"100%", objectFit: "contain"}}
                />
            )
        }
    }

    render() {
        return (
            <Container style={{height:"100%"}}>
                <Col className="d-flex flex-column" style={{height: '100%', paddingLeft: "6px", paddingRight: "6px"}}>
                    <Row className="d-flex flex-row">
                        <h5>Data visualization</h5>
                    </Row>
                    <Row className="d-flex flex-row" style={{flexGrow:'1', overflowY: "auto", height:"100%"}}>
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
                                           key="switch_show_model_prediction"
                                           id="switch_show_model_prediction"
                                           onChange={() => this.props.onShowModelPredictionSwitchChange()}
                                           checked={this.props.show_model_prediction}
                                           style={{marginRight: "10px"}}
                                    />
                                    <label className="form-check-label" htmlFor="switch_show_model_prediction">
                                        Show model prediction
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