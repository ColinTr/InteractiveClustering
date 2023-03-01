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
        if(this.props.search_filtered_list != null) {
            return (
                <Container>
                    {this.props.search_filtered_list.map((feature) => (
                        <Form.Check
                            type="checkbox"
                            id={"checkbox" + feature.name}
                            label={"Feature " + feature.name}
                            key={feature.name}
                            onChange={() => this.props.onChangeCheckbox(feature.index)}
                            checked={feature.checked}
                        />
                    ))}
                </Container>
            )
        } else {
            return (
                <Container>
                    <i>No features to display...</i>
                </Container>
            )
        }
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
                                value={this.props.search_query}
                                onChange={this.props.onChangeSearch}
                                sx={{ input: { color: 'white'} }}
                            />
                        </Col>
                    </Row>

                    <Row style={{flexDirection:"row", paddingTop: "10px"}}>
                        <Col style={{paddingLeft:12, paddingRight:5}} className="d-flex flex-column justify-content-end">
                            <button type="button" className="btn btn-secondary"
                                    style={{paddingTop:0, paddingBottom:0, paddingLeft:"5px", paddingRight:"5px", fontSize: "small"}}
                                    onClick={() => this.props.onCheckButtonClick()}>
                                Check all
                            </button>
                        </Col>
                        <Col style={{paddingLeft:5, paddingRight:5}} className="d-flex flex-column justify-content-end">
                            <button type="button" className="btn btn-secondary"
                                    style={{paddingTop:0, paddingBottom:0, paddingLeft:"5px", paddingRight:"5px", fontSize: "small"}}
                                    onClick={() => this.props.onUncheckButtonClick()}>
                                Uncheck all
                            </button>
                        </Col>
                        <Col style={{paddingLeft:5, paddingRight:12}} className="d-flex flex-column justify-content-end">
                            <button type="button" className="btn btn-secondary"
                                    style={{paddingTop:0, paddingBottom:0, paddingLeft:"5px", paddingRight:"5px", fontSize: "small"}}
                                    onClick={() => this.props.onClearSearchButtonClick()}>
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