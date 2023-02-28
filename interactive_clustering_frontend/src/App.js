import './App.css';
import Col from "react-bootstrap/Col";
import Row from "react-bootstrap/Row";
import Container from "react-bootstrap/Container";
import DatasetSelector from "./components/DatasetSelector";
import FeatureSelection from "./components/FeatureSelection";
import DataVisualization from "./components/DataVisualization";
import AgglomerativeClustering from "./components/AgglomerativeClustering";
import ModelSelection from "./components/ModelSelection";


document.body.style.backgroundColor = "#22223b";

function App() {
  return (
    <div className="App">
        <Container style={{height: '100vh'}}>
            <Row style={{height: '100%'}} className="d-flex flex-row justify-content-center align-items-center">

                <Col className="col-lg-3 col-12 d-flex flex-column" style={{height: "80vh"}}>
                    <Row className="my_row py-2 d-flex flex-row" style={{flexGrow:'1'}}>
                        <ModelSelection/>
                    </Row>
                    <Row className="my_row py-1 d-flex flex-row">
                        <AgglomerativeClustering/>
                    </Row>
                </Col>

                <Col className="col-lg-6 col-12 d-flex flex-column justify-content-center" style={{height: "80vh"}}>
                    <Row className="my_row mx-1 py-2 d-flex flex-row" style={{flexGrow:'1'}}>
                        <DataVisualization/>
                    </Row>
                    <Row className="my_row mx-1 py-2">
                        <DatasetSelector />
                    </Row>
                </Col>

                <Col className="col-lg-3 col-12 d-flex flex-column" style={{height: "80vh"}}>
                    <Row className="my_row py-2" style={{flexGrow:'1'}}>
                        <FeatureSelection/>
                    </Row>
                </Col>

            </Row>
        </Container>
    </div>
  );
}

export default App;
