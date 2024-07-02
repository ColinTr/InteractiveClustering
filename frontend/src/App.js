/*
Software Name : InteractiveClustering
Version: 1.0
SPDX-FileCopyrightText: Copyright (c) 2024 Orange
SPDX-License-Identifier: MIT
This software is distributed under the MIT License,
the text of which is available at https://spdx.org/licenses/MIT.html
or see the "license.txt" file for more details.
*/

import './App.css';
import FullPage from "./components/FullPage";
import {SnackbarProvider} from "notistack";
import React from "react";

document.body.style.backgroundColor = "#22223b";

function App() {
  return (
    <div className="App d-flex justify-content-center align-items-center" style={{height: "100vh"}}>
        <SnackbarProvider />
        <FullPage />
    </div>
  );
}

export default App;
