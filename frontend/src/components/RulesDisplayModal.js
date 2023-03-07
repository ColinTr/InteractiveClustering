import React from 'react';
import Button from 'react-bootstrap/Button';
import Modal from 'react-bootstrap/Modal';
import Row from "react-bootstrap/Row";

function formatRulesText(json_response, training_mode) {
    if(training_mode === "multi_class"){
        return (<pre id="json"> {json_response} </pre>)
    }
    if(training_mode === "one_vs_rest"){
        const arr = [];
        Object.keys(json_response).forEach((key) => {
            arr.push(
                <Row key={"row_" + key}>
                    {key}
                    <pre id="json"> {json_response[key]} </pre>
                    <hr />
                </Row>
            )
        })
        return (arr)
    }

    return null
}

const RulesDisplayModal = (props) => {
    return (
        <Modal show={props.rules_modal_is_open} onHide={props.closeRulesModal}>
            <Modal.Header closeButton>
                <Modal.Title>Rules for the last clustering run</Modal.Title>
            </Modal.Header>
            <Modal.Body>

                <div>This model had {(props.decision_tree_response_accuracy_score * 100).toFixed(2)}% training accuracy.</div>

                <hr />

                <iframe title="pdf" width="100%" height="600px" srcDoc={props.decision_tree_response_pdf_file}></iframe>

                {formatRulesText(props.decision_tree_response_text_rules, props.decision_tree_response_training_mode)}
            </Modal.Body>
            <Modal.Footer>
                <Button variant="secondary" onClick={props.closeRulesModal}>
                    Close
                </Button>
            </Modal.Footer>
        </Modal>
    );
}

export default RulesDisplayModal;