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
                     style={{width: "95%", height: "100%", objectFit: "contain"}}
                />
            )
        }
    }

    render() {
        return (
            <Container>
                <Col className="d-flex flex-column" style={{height: '100%', paddingLeft: "6px", paddingRight: "6px"}}>
                    <Row className="d-flex flex-row">
                        <h5>Data visualization</h5>
                    </Row>
                    <Row className="d-flex flex-row" style={{flexGrow:'1', overflowY: "auto"}}>
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
                            <Row className="d-flex flex-row">
                                Color with the...
                            </Row>
                            <Row className="d-flex flex-row">
                                <div style={{display: "flex"}}>
                                    <div className="form-check" style={{marginRight: "10px"}}>
                                        <input className="form-check-input" type="radio" name="flexRadio" id="flexRadioDefault1"
                                               checked
                                               onChange={this.props.onGroundTruthRadioButtonChange}
                                               disabled={this.props.ground_truth_radio_button_disabled}
                                               title="tooltip on radio!"/>
                                        <label className="form-check-label" htmlFor="flexRadioDefault1">
                                            Ground truth
                                        </label>
                                    </div>
                                    <div className="form-check">
                                        <input className="form-check-input" type="radio" name="flexRadio" id="flexRadioDefault2"
                                               onChange={this.props.onPredictionRadioButtonChange}
                                               disabled={this.props.prediction_radio_button_disabled} />
                                        <label className="form-check-label" htmlFor="flexRadioDefault2">
                                            Prediction
                                        </label>
                                    </div>
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