import React from "react";
import Container from "react-bootstrap/Container";
import {FlexibleXYPlot, MarkSeries, XAxis, YAxis} from "react-vis";
import Row from "react-bootstrap/Row";
import Col from "react-bootstrap/Col";
import {AutoSizer} from "react-virtualized";


class DataVisualization extends React.Component {
    constructor(props) {
        super(props);
        this.state = {
            index : null
        };
    }

    render() {
        const {index} = this.state;

        const data = [
            {x: 0, y: 8, color: 0 === index ? 1 : 0},
            {x: 1, y: 5, color: 1 === index ? 1 : 0},
            {x: 2, y: 4, color: 2 === index ? 1 : 0},
            {x: 3, y: 9, color: 3 === index ? 1 : 0},
            {x: 4, y: 1, color: 4 === index ? 1 : 0},
            {x: 5, y: 7, color: 5 === index ? 1 : 0},
            {x: 6, y: 6, color: 6 === index ? 1 : 0},
            {x: 7, y: 3, color: 7 === index ? 1 : 0},
            {x: 8, y: 2, color: 8 === index ? 1 : 0},
            {x: 9, y: 0, color: 9 === index ? 1 : 0}
        ]

        return (
            <Container className="d-flex">
                <Col className="d-flex flex-column">
                    <Row className="d-flex flex-row">
                        <h5>Data visualization</h5>
                    </Row>
                    <Row className="d-flex flex-row flex-grow" style={{flexGrow:'1', padding: "10px"}}>
                        <div style={{ marginRight:"10px", backgroundColor: 'white'}}>
                            <AutoSizer>
                                {({ height, width }) => (
                                    <FlexibleXYPlot
                                        width={width}
                                        height={height}
                                        colorDomain={[0, 1]}
                                        onMouseLeave={() => this.setState({index: null})}
                                    >
                                        <XAxis />
                                        <YAxis />
                                        <MarkSeries
                                            data={data}
                                            stroke="white"
                                            onNearestXY={(datapoint, {index}) => this.setState({index})}
                                            onSeriesClick={(event)=>{
                                                console.log("clicked on point " + index)
                                            }}
                                        />
                                    </FlexibleXYPlot>
                                )}
                            </AutoSizer>
                        </div>
                    </Row>
                </Col>
            </Container>
        )
    }
}

export default DataVisualization;