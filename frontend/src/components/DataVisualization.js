import React from "react";
import Container from "react-bootstrap/Container";
import Row from "react-bootstrap/Row";
import Col from "react-bootstrap/Col";
import Button from "react-bootstrap/Button";


class DataVisualization extends React.Component {
    render() {
        return (
            <Container>
                <Col className="d-flex flex-column" style={{height: '100%', paddingLeft: "6px", paddingRight: "6px"}}>
                    <Row className="d-flex flex-row">
                        <h5>Data visualization</h5>
                    </Row>
                    <Row className="d-flex flex-row" style={{flexGrow:'1', overflowY: "auto"}}>
                        <i>Content row...</i>
                    </Row>
                    <Row className="d-flex flex-row" style={{paddingLeft: "6px", paddingRight: "6px"}}>
                        <Col className="d-flex flex-column">
                            <Row>
                                Get T-SNE of the...
                            </Row>
                            <Row>
                                <Col>
                                    <Button onClick={() => this.props.onRawDataButtonClick()}>
                                        Raw data
                                    </Button>
                                </Col>
                                <Col>
                                    <Button onClick={() => this.props.onProjectionButtonClick()}>
                                        Projection
                                    </Button>
                                </Col>
                            </Row>
                        </Col>
                        <Col className="d-flex flex-column" >
                            <Row>
                                Color with the...
                            </Row>
                            <Row>
                                <Col>
                                    <div className="form-check">
                                        <input className="form-check-input" type="radio" name="flexRadio" id="flexRadioDefault1"
                                               checked
                                               onChange={this.props.onGroundTruthRadioButtonChange}
                                               disabled={this.props.ground_truth_radio_button_disabled}
                                               title="tooltip on radio!"/>
                                        <label className="form-check-label" htmlFor="flexRadioDefault1">
                                            Ground truth
                                        </label>
                                    </div>
                                </Col>
                                <Col>
                                    <div className="form-check">
                                        <input className="form-check-input" type="radio" name="flexRadio" id="flexRadioDefault2"
                                               onChange={this.props.onPredictionRadioButtonChange}
                                               disabled={this.props.prediction_radio_button_disabled} />
                                        <label className="form-check-label" htmlFor="flexRadioDefault2">
                                            Prediction
                                        </label>
                                    </div>
                                </Col>
                            </Row>
                        </Col>
                    </Row>
                </Col>
            </Container>
        )
    }
}

export default DataVisualization;