import React from "react";
import Container from "react-bootstrap/Container";
import TextField from "@mui/material/TextField";
import { styled } from "@mui/material/styles";
import Row from "react-bootstrap/Row";
import Col from "react-bootstrap/Col";
import Form from "react-bootstrap/Form";

const StyledTextField = styled(TextField)({
    "& label": { color: "white" },
    "&:hover label": { fontWeight: 700 },
    "& label.Mui-focused": { color: "white" },
    "& .MuiInput-underline:after": { borderBottomColor: "white" },
    "& .MuiOutlinedInput-root": {
        "& fieldset": { borderColor: "white" },
        "&:hover fieldset": { borderColor: "white", borderWidth: 2 },
        "&.Mui-focused fieldset": { borderColor: "white" }
    }
});

class FeatureSelection extends React.Component {
    features_list = () => {
        if(this.props.search_filtered_features_list != null) {
            return (
                <Container>
                    {this.props.search_filtered_features_list.map((feature) => (
                        <Row className="d-flex flex-row" key={"row_" + feature.name}>
                            <Col className="d-flex flex-column col-1">
                                <div className="form-check">
                                    <input type="radio"
                                           className="form-check-input"
                                           onChange={() => this.props.onFeatureRadioButtonChange(feature.name)}
                                           name="classFeatureRadio"
                                           key={"class_radio_" + feature.name}
                                    />
                                </div>
                            </Col>
                            <Col className="d-flex flex-column col-10">
                                <Form.Check type="checkbox"
                                            onChange={() => this.props.onChangeCheckbox(feature.index)}
                                            checked={feature.checked}
                                            key={"checkbox_" + feature.name}
                                            disabled={feature.disabled}
                                            label={feature.name}
                                />
                            </Col>
                        </Row>
                    ))}
                </Container>
            )
        } else {
            return (
                <Container>
                    <i>Load a dataset to start exploring features</i>
                </Container>
            )
        }
    }

    values_list = () => {
        if(this.props.search_filtered_unique_values_list != null) {
            return (
                <Container>
                    {this.props.search_filtered_unique_values_list.map((feature) => (
                        <div className="form-check form-switch" key={"div_switch_" + feature.name}>
                            <input className="form-check-input"
                                   type="checkbox"
                                   key={"switch_" + feature.name}
                                   id={"switch_" + feature.name}
                                   onChange={() => this.props.onUniqueValueSwitchChange(feature.index)}
                                   checked={feature.checked}
                            />
                            <label className="form-check-label" htmlFor={"switch_" + feature.name}>
                                {feature.name}
                            </label>
                        </div>
                    ))}
                </Container>
            )
        } else {
            if(this.props.search_filtered_features_list == null){
                return (
                    <Container>
                        <i>Load a dataset to start exploring class values</i>
                    </Container>
                )
            } else {
                return (
                    <Container>
                        <i>Select one of the features as class</i>
                    </Container>
                )
            }
        }
    }

    render() {
        return (
            <Container>
                <Col className="d-flex flex-column" style={{height: '100%', paddingLeft: "6px", paddingRight: "6px"}}>

                    <Row className="d-flex flex-row" style={{paddingBottom: "5px"}}>
                        <h5>Feature selection</h5>
                    </Row>

                    <Row className="d-flex flex-row">
                        <Col>
                            <StyledTextField
                                id="outlined-basic1"
                                variant="outlined"
                                fullWidth
                                label="Search"
                                value={this.props.feature_search_query}
                                onChange={this.props.onChangeFeaturesSearch}
                                sx={{ input: { color: 'white'} }}
                            />
                        </Col>
                    </Row>

                    <Row style={{flexDirection:"row", paddingTop: "10px", paddingBottom: "10px"}}>
                        <Col style={{paddingLeft:12, paddingRight:5}} className="d-flex flex-column justify-content-end">
                            <button type="button" className="btn btn-secondary"
                                    style={{paddingTop:0, paddingBottom:0, paddingLeft:"5px", paddingRight:"5px", fontSize: "small"}}
                                    onClick={() => this.props.onCheckAllButtonClick()}>
                                Check all
                            </button>
                        </Col>
                        <Col style={{paddingLeft:5, paddingRight:5}} className="d-flex flex-column justify-content-end">
                            <button type="button" className="btn btn-secondary"
                                    style={{paddingTop:0, paddingBottom:0, paddingLeft:"5px", paddingRight:"5px", fontSize: "small"}}
                                    onClick={() => this.props.onUncheckAllButtonClick()}>
                                Uncheck all
                            </button>
                        </Col>
                        <Col style={{paddingLeft:5, paddingRight:12}} className="d-flex flex-column justify-content-end">
                            <button type="button" className="btn btn-secondary"
                                    style={{paddingTop:0, paddingBottom:0, paddingLeft:"5px", paddingRight:"5px", fontSize: "small"}}
                                    onClick={() => this.props.onClearFeaturesSearchButtonClick()}>
                                Clear search
                            </button>
                        </Col>
                    </Row>

                    <Row className="d-flex flex-row" style={{overflowY: "auto", flex:"1 1 auto", height: '0px'}}>
                        {this.features_list()}
                    </Row>

                    <hr/>

                    <Row className="d-flex flex-row" style={{paddingBottom: "5px"}}>
                        <h5>Class modalities</h5>
                    </Row>

                    <Row className="d-flex flex-row">
                        <Col>
                            <Row className="d-flex flex-row">
                                <Col>
                                    <StyledTextField
                                        id="outlined-basic2"
                                        variant="outlined"
                                        fullWidth
                                        label="Search"
                                        value={this.props.unique_values_search_query}
                                        onChange={this.props.onChangeUniqueValuesSearch}
                                        sx={{ input: { color: 'white'} }}
                                    />
                                </Col>
                            </Row>
                        </Col>
                    </Row>

                    <Row style={{flexDirection:"row", paddingTop: "10px", paddingBottom: "10px"}}>
                        <Col style={{paddingLeft:12, paddingRight:5}} className="d-flex flex-column justify-content-end">
                            <button type="button" className="btn btn-secondary"
                                    style={{paddingTop:0, paddingBottom:0, paddingLeft:"5px", paddingRight:"5px", fontSize: "small"}}
                                    onClick={() => this.props.onSwitchAllOnButtonClick()}>
                                Switch all on
                            </button>
                        </Col>
                        <Col style={{paddingLeft:5, paddingRight:5}} className="d-flex flex-column justify-content-end">
                            <button type="button" className="btn btn-secondary"
                                    style={{paddingTop:0, paddingBottom:0, paddingLeft:"5px", paddingRight:"5px", fontSize: "small"}}
                                    onClick={() => this.props.onSwitchAllOffButtonClick()}>
                                Switch all off
                            </button>
                        </Col>
                        <Col style={{paddingLeft:5, paddingRight:12}} className="d-flex flex-column justify-content-end">
                            <button type="button" className="btn btn-secondary"
                                    style={{paddingTop:0, paddingBottom:0, paddingLeft:"5px", paddingRight:"5px", fontSize: "small"}}
                                    onClick={() => this.props.onClearUniqueValuesSearchButtonClick()}>
                                Clear search
                            </button>
                        </Col>
                    </Row>

                    <Row className="d-flex flex-row" style={{overflowY: "auto", flex:"1 1 auto", height: '0px'}}>
                        {this.values_list()}
                    </Row>

                </Col>
            </Container>
        )
    }
}

export default FeatureSelection;