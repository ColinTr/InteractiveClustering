import React from "react";
import Container from "react-bootstrap/Container";
import TextField from "@mui/material/TextField";
import { styled } from "@mui/material/styles";
import Form from 'react-bootstrap/Form';
import Row from "react-bootstrap/Row";
import Col from "react-bootstrap/Col";


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
            search_query: '',
            search_filtered_list : null,
            formatted_features : mock_features.map((feature, index) => ({"name": feature.feature_name, "checked" : true, index : index}))
        };

        this.state.search_filtered_list =  this.state.formatted_features
    }

    onChangeSearch = query => {
        if (query.target.value !== '') {
            this.setState({ search_query: query.target.value });
        }

        const updated_filtered_list = this.getUpdatedFilteredList(this.state.formatted_features, query.target.value)
        this.setState({ search_filtered_list: updated_filtered_list })
    };

    getUpdatedFilteredList = (features_list, query) => {
        return features_list.filter((feature) => {
            if (query === "") {
                return features_list
            } else {
                return feature.name.toLowerCase().includes(query.toLowerCase())
            }
        })
    }

    onChangeCheckbox = i => {
        let formatted_features = [...this.state.formatted_features];  // Make a shallow copy
        let formatted_features_item = {...formatted_features[i]};  // Get the element we want to update
        formatted_features_item.checked = !formatted_features_item.checked  // Change it
        formatted_features[i] = formatted_features_item // Replace it in the array's copy

        const updated_filtered_list = this.getUpdatedFilteredList(formatted_features, this.state.search_query)
        this.setState({formatted_features: formatted_features, search_filtered_list: updated_filtered_list})  // And finally replace the array in the state
    }

    onUncheckButtonClick = () => {
        let formatted_features = [...this.state.formatted_features];  // Make a shallow copy
        this.state.search_filtered_list.map((feature) => (  // For each feature currently displayed...
            formatted_features[feature.index].checked = false
        ))

        const updated_filtered_list = this.getUpdatedFilteredList(formatted_features, this.state.search_query)
        this.setState({formatted_features: formatted_features, search_filtered_list: updated_filtered_list})  // And finally replace the array in the state
    }

    onCheckButtonClick = () => {
        let formatted_features = [...this.state.formatted_features];  // Make a shallow copy
        this.state.search_filtered_list.map((feature) => (  // For each feature currently displayed...
            formatted_features[feature.index].checked = true
        ))

        const updated_filtered_list = this.getUpdatedFilteredList(formatted_features, this.state.search_query)
        this.setState({formatted_features: formatted_features, search_filtered_list: updated_filtered_list})  // And finally replace the array in the state
    }

    onClearSearchButtonClick = () => {
        const updated_filtered_list = this.getUpdatedFilteredList(this.state.formatted_features, '')
        this.setState({search_filtered_list: updated_filtered_list, search_query: ''})  // And finally replace the array in the state
    }

    features_list = () => {
        return (
            <Container>
                {this.state.search_filtered_list.map((feature) => (
                    <Form.Check
                        type="checkbox"
                        id={"checkbox" + feature.name}
                        label={"Feature " + feature.name}
                        key={feature.name}
                        onChange={() => this.onChangeCheckbox(feature.index)}
                        checked={feature.checked}
                    />
                ))}
            </Container>
        );
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
                                value={this.state.search_query}
                                onChange={this.onChangeSearch}
                                sx={{ input: { color: 'white'} }}
                            />
                        </Col>
                    </Row>

                    <Row style={{flexDirection:"row", paddingTop: "10px"}}>
                        <Col style={{paddingLeft:12, paddingRight:5}} className="d-flex flex-column justify-content-end">
                            <button type="button" className="btn btn-secondary" style={{paddingTop:0, paddingBottom:0, paddingLeft:"5px", paddingRight:"5px", fontSize: "small"}} onClick={() => this.onCheckButtonClick()}>
                                Check all
                            </button>
                        </Col>
                        <Col style={{paddingLeft:5, paddingRight:5}} className="d-flex flex-column justify-content-end">
                            <button type="button" className="btn btn-secondary" style={{paddingTop:0, paddingBottom:0, paddingLeft:"5px", paddingRight:"5px", fontSize: "small"}} onClick={() => this.onUncheckButtonClick()}>
                                Uncheck all
                            </button>
                        </Col>
                        <Col style={{paddingLeft:5, paddingRight:12}} className="d-flex flex-column justify-content-end">
                            <button type="button" className="btn btn-secondary" style={{paddingTop:0, paddingBottom:0, paddingLeft:"5px", paddingRight:"5px", fontSize: "small"}} onClick={() => this.onClearSearchButtonClick()}>
                                Clear search
                            </button>
                        </Col>
                    </Row>

                    <hr/>

                    <Row className="d-flex flex-row" style={{overflowY: "auto", flex:"1 1 auto", height: '0px'}}>
                        {this.features_list()}
                    </Row>

                </Col>
            </Container>
        )
    }
}

export default FeatureSelection;