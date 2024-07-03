/*
Software Name : InteractiveClustering
Version: 1.0
SPDX-FileCopyrightText: Copyright (c) 2024 Orange
SPDX-License-Identifier: MIT
This software is distributed under the MIT License,
the text of which is available at https://spdx.org/licenses/MIT.html
or see the "license.txt" file for more details.
*/

import React, { forwardRef } from "react"
import { SnackbarContent } from "notistack"
import Container from "react-bootstrap/Container";
import Row from "react-bootstrap/Row";
import Col from "react-bootstrap/Col";
import { FontAwesomeIcon } from "@fortawesome/react-fontawesome";
import { faSpinner } from "@fortawesome/free-solid-svg-icons";

const WaitSnackbar = forwardRef((props, ref) => {
    return (
        <SnackbarContent ref={ref}>
            <Container style={{backgroundColor: "white", color: "black", borderRadius: "5px", paddingLeft: "10px", paddingRight: "10px", paddingBottom: "10px", paddingTop: "5px"}}>
                <Row>
                    <Col className="col-auto d-flex flex-column" style={{justifyContent: "center"}}>
                        <FontAwesomeIcon icon={faSpinner} spin />
                    </Col>
                    <Col className="col-auto d-flex flex-column" style={{justifyContent: "center", flexGrow:'1', textAlign: "left"}}>
                        {props.message}
                    </Col>
                </Row>
            </Container>
        </SnackbarContent>
    )
})

export default WaitSnackbar;