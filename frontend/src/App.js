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
