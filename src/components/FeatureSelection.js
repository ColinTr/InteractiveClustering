import React from "react";
import Container from "react-bootstrap/Container";
import TextField from "@mui/material/TextField";
import { styled } from "@mui/material/styles";
import Form from 'react-bootstrap/Form';
import Row from "react-bootstrap/Row";
import Col from "react-bootstrap/Col";
import {map} from "react-bootstrap/ElementChildren";


const StyledTextField = styled(TextField)({
    "& label": {
        color: "white"
    },
    "&:hover label": {
        fontWeight: 700
    },
    "& label.Mui-focused": {
        color: "white"
    },
    "& .MuiInput-underline:after": {
        borderBottomColor: "white"
    },
    "& .MuiOutlinedInput-root": {
        "& fieldset": {
            borderColor: "white"
        },
        "&:hover fieldset": {
            borderColor: "white",
            borderWidth: 2
        },
        "&.Mui-focused fieldset": {
            borderColor: "white"
        }
    }
});

const mock_features = [
    {feature_name: "f1"}, {feature_name: "f2"},
    {feature_name: "f3"}, {feature_name: "f4"},
    {feature_name: "f5"}, {feature_name: "f6"},
    {feature_name: "f7"}, {feature_name: "f8"},
    {feature_name: "f9"}, {feature_name: "f10"},
    {feature_name: "f11"}, {feature_name: "f12"},
    {feature_name: "f13"}, {feature_name: "f14"},
    {feature_name: "f15"}, {feature_name: "f16"},
    {feature_name: "f17"}, {feature_name: "f18"},
    {feature_name: "f19"}, {feature_name: "f20"},
    {feature_name: "f21"}, {feature_name: "f22"},
    {feature_name: "f23"}, {feature_name: "f24"},
    {feature_name: "f25"}, {feature_name: "f26"},
    {feature_name: "f27"}, {feature_name: "f28"},
    {feature_name: "f29"}, {feature_name: "f30"},
    {feature_name: "f31"}, {feature_name: "f32"},
    {feature_name: "f33"}
];

class FeatureSelection extends React.Component {
    constructor(props) {
        super(props);
        this.state = {
            searchQuery: null,
            formatted_features : mock_features.map((feature, index) => ({"name": feature.feature_name, "checked" : true, index : index}))
        };
    }

    onChangeSearch = query => {
        this.setState({ searchQuery: query });
    };

    onChangeCheckbox = i => {
        let formatted_features = [...this.state.formatted_features];  // Make a shallow copy
        let formatted_features_item = {...formatted_features[i]};  // Get the element we want to update
        formatted_features_item.checked = !formatted_features_item.checked  // Change it
        formatted_features[i] = formatted_features_item // Replace it in the array's copy
        this.setState({formatted_features})  // And finally replace the array in the state
    }

    features_list = props => {
        return (
            <Container>
                {this.state.formatted_features.map((feature, i) => (
                    <Form.Check
                        type="checkbox"
                        id={"checkbox" + feature.name}
                        label={"Feature " + feature.name}
                        key={feature.name}
                        onChange={() => this.onChangeCheckbox(i)}
                        checked={this.state.formatted_features[feature.index].checked}
                    />
                ))}
            </Container>
        );
    }

    onUncheckButtonClick = () => {
        let formatted_features = [...this.state.formatted_features];  // Make a shallow copy
        formatted_features.map((feature, i) => (
            feature.checked = false
        ))
        this.setState({formatted_features})  // And finally replace the array in the state
    }

    onCheckButtonClick = () => {
        let formatted_features = [...this.state.formatted_features];  // Make a shallow copy
        formatted_features.map((feature, i) => (
            feature.checked = true
        ))
        this.setState({formatted_features})  // And finally replace the array in the state
    }

    render() {
        return (
            <Container>
                <Col className="d-flex flex-column" style={{height: '100%', paddingLeft: "6px", paddingRight: "6px"}}>

                    <Row className="d-flex flex-row">
                        <h5>Feature selection</h5>
                    </Row>

                    <Row className="d-flex flex-row" style={{paddingTop: "10px"}}>
                        <Col>
                            <StyledTextField
                                id="outlined-basic"
                                variant="outlined"
                                fullWidth
                                label="Search"
                                sx={{ input: { color: 'white'} }}
                            />
                        </Col>
                    </Row>

                    <Row style={{flexDirection:"row", paddingTop: "10px"}}>
                        <Col>
                            <button type="button" className="btn btn-secondary " style={{paddingTop:0, paddingBottom:0, paddingLeft:"5px", paddingRight:"5px"}} onClick={() => this.onCheckButtonClick()}>
                                Check all
                            </button>
                        </Col>
                        <Col>
                            <button type="button" className="btn btn-secondary" style={{paddingTop:0, paddingBottom:0, paddingLeft:"5px", paddingRight:"5px"}} onClick={() => this.onUncheckButtonClick()}>
                                Uncheck all
                            </button>
                        </Col>
                    </Row>

                    <Row className="d-flex flex-row" style={{overflowY: "auto", flex:"1 1 auto", height: '0px', paddingTop: "10px"}}>
                        {this.features_list()}
                    </Row>

                </Col>
            </Container>
        )
    }
}

export default FeatureSelection;