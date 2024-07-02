/*
Software Name : InteractiveClustering
Version: 1.0
SPDX-FileCopyrightText: Copyright (c) 2024 Orange
SPDX-License-Identifier: MIT
This software is distributed under the MIT License,
the text of which is available at https://spdx.org/licenses/MIT.html
or see the "license.txt" file for more details.
*/

import React from 'react';
import Button from 'react-bootstrap/Button';
import Modal from 'react-bootstrap/Modal';
import Table from 'react-bootstrap/Table';

function formatFeature(point_feature_names, point_feature_values) {
    if(point_feature_names !== null) {
        const arr = []
        Object.keys(point_feature_names).forEach((name, index) => {
            arr.push(
                <tr key={"row_" + index}>
                    <td style={{wordWrap: "break-word", maxWidth: "40vw"}}>
                        {point_feature_names[index]}
                    </td>
                    <td style={{wordWrap: "break-word", maxWidth: "40vw"}}>
                        {point_feature_values[0][index]}
                    </td>
                </tr>
            )
        })
        return arr
    }
}

const FeatureDisplayModal = (props) => {
    return (
        <Modal show={props.feature_modal_is_open} onHide={props.closeFeaturesModal} dialogClassName="feature-modal">
            <Modal.Header closeButton>
                <Modal.Title>Values of point</Modal.Title>
            </Modal.Header>
            <Modal.Body>
                <Table striped bordered hover>
                    <thead>
                        <tr>
                            <th>Feature name</th>
                            <th>Value</th>
                        </tr>
                    </thead>
                    <tbody>
                        {formatFeature(props.point_feature_names, props.point_feature_values)}
                    </tbody>
                </Table>
            </Modal.Body>
            <Modal.Footer>
                <Button variant="secondary" onClick={props.closeFeaturesModal}>
                    Close
                </Button>
            </Modal.Footer>
        </Modal>
    )
}

export default FeatureDisplayModal;