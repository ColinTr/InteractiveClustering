import './App.css';
import FullPage from "./components/FullPage";
import {SnackbarProvider} from "notistack";
import React from "react";

document.body.style.backgroundColor = "#22223b";

function App() {
  return (
    <div className="App">
        <SnackbarProvider />
        <FullPage />
    </div>
  );
}

export default App;
