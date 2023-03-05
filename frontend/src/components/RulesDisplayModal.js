import React from 'react';
import Button from 'react-bootstrap/Button';
import Modal from 'react-bootstrap/Modal';

const RulesDisplayModal = (props) => {
    return (
        <Modal show={props.rules_modal_is_open} onHide={props.closeRulesModal}>
            <Modal.Header closeButton>
                <Modal.Title>Rules for the last clustering run</Modal.Title>
            </Modal.Header>
            <Modal.Body>
                <pre id="json">
                    {props.text_rules}
                </pre>
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