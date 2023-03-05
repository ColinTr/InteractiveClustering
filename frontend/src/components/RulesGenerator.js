import React from "react";
import Container from "react-bootstrap/Container";
import Row from "react-bootstrap/Row";
import Col from "react-bootstrap/Col";
import Button from "react-bootstrap/Button";
import {AiOutlineQuestionCircle} from "react-icons/ai";
import {Tooltip} from "@mui/material";
import Form from 'react-bootstrap/Form';

class RulesGenerator extends React.Component {

    render() {
        return (
            <Container>
                <Col className="d-flex flex-column" style={{height: '100%', paddingLeft: "6px", paddingRight: "6px"}}>
                    <Row className="d-flex flex-row pb-1">
                        <h5>Rules generation</h5>
                    </Row>
                    <Row className="d-flex flex-row">
                        <div style={{display:"flex", flexDirection:"row"}}>
                            <Form.Check inline checked={this.props.decision_tree_training_mode === "multi_class"} value="multi_class" onChange={(e) => this.props.onDecisionTreeRadioButtonChange(e.target.value)} name="decision_tree_group" type="radio" id="multi_class_radio_button_1" style={{marginRight: "5px"}}/>
                            <Form.Label htmlFor="multi_class_radio_button_1">
                                <Tooltip title="Rules are formed by training a single decision tree for all the clusters">
                                    <div style={{display: "flex", alignItems: "center"}}>
                                        Multi-Class <AiOutlineQuestionCircle style={{marginLeft: "5px"}}/>
                                    </div>
                                </Tooltip>
                            </Form.Label>

                            <Form.Check inline checked={this.props.decision_tree_training_mode === "one_vs_rest"} value="one_vs_rest" onChange={(e) => this.props.onDecisionTreeRadioButtonChange(e.target.value)} name="decision_tree_group" type="radio" id="multi_class_radio_button_2" style={{marginLeft:"15px", marginRight: "5px"}}/>
                            <Form.Label htmlFor="multi_class_radio_button_2">
                                <Tooltip title="Rules are formed by training a decision tree for each cluster">
                                    <div style={{display: "flex", alignItems: "center"}}>
                                        One-Vs-Rest <AiOutlineQuestionCircle  style={{marginLeft: "5px"}}/>
                                    </div>
                                </Tooltip>
                            </Form.Label>
                        </div>
                    </Row>

                    <Row className="d-flex flex-row" style={{marginBottom: "10px"}}>
                        <Col className="col-10 d-flex flex-column">
                            <Tooltip title="Generate rules for all the classes (known + unknown), or only for the unknown classes">
                                <label className="form-check-label" htmlFor="switch_train_unknown_ony">
                                    Unknown classes only <AiOutlineQuestionCircle style={{marginLeft: "5px"}}/>
                                </label>
                            </Tooltip>
                        </Col>
                        <Col className="col-2 d-flex flex-column">
                            <div className="form-check form-switch justify-content-center" style={{display: "flex", alignItems: "center"}}>
                                <input className="form-check-input"
                                       type="checkbox"
                                       key="switch_train_unknown_ony"
                                       id="switch_train_unknown_ony"
                                       onChange={this.props.onRulesUnknownClassesOnlySwitchChange}
                                       style={{marginRight: "10px"}}
                                />
                            </div>
                        </Col>
                    </Row>

                    <Row className="d-flex flex-row">
                        <Col className="col-8 d-flex flex-column">
                            <Tooltip title="The maximum depth of the tree. If empty, then nodes are expanded until all leaves are pure or until all leaves contain less than min_samples_split samples.">
                                <p style={{display: "flex", alignItems: "center"}}>
                                    Max depth <AiOutlineQuestionCircle  style={{marginLeft: "5px"}}/>
                                </p>
                            </Tooltip>
                        </Col>
                        <Col className="col-4 d-flex flex-column">
                            <input type="number"
                                   min={0}
                                   placeholder="∅"
                                   step={1}
                                   defaultValue={this.props.decision_tree_max_depth}
                                   onChange={this.props.on_decision_tree_max_depth_change}
                            />
                        </Col>
                    </Row>

                    <Row className="d-flex flex-row">
                        <Col className="col-8 d-flex flex-column">
                            <Tooltip title="The minimum number of samples required to split an internal node:">
                                <p style={{display: "flex", alignItems: "center"}}>
                                    Min samples split <AiOutlineQuestionCircle  style={{marginLeft: "5px"}}/>
                                </p>
                            </Tooltip>
                        </Col>
                        <Col className="col-4 d-flex flex-column">
                            <input type="number"
                                   min={0}
                                   placeholder="Max depth"
                                   step={1}
                                   defaultValue={this.props.decision_tree_min_samples_split}
                                   onChange={this.props.on_decision_tree_min_samples_split_change}
                            />
                        </Col>
                    </Row>
                    <Row className="d-flex flex-row">
                        <Col className="col-6 d-flex flex-column">
                            <Button style={{width:'120px'}} onClick={() => this.props.onRulesRunButtonClick()}>
                                Run
                            </Button>
                        </Col>
                        <Col className="col-6 d-flex flex-column flex-column align-items-end">
                            <Button variant="success" style={{width:'120px'}} onClick={() => this.props.onShowRulesButtonClick()}>
                                Show rules
                            </Button>
                        </Col>
                    </Row>
                </Col>
            </Container>
        )
    }
}

export default RulesGenerator;