import React from 'react';
import Row from "react-bootstrap/Row";
import Col from "react-bootstrap/Col";
import ModelSelection from "./ModelSelection";
import AgglomerativeClustering from "./AgglomerativeClustering";
import DataVisualization from "./DataVisualization";
import DatasetSelector from "./DatasetSelector";
import FeatureSelection from "./FeatureSelection";
import Container from "react-bootstrap/Container";


class FullPage extends React.Component {

    constructor(props) {
        super(props);
        this.state = {
            // The states of the children that need to be shared are "lifted" here
            formatted_features : null,  // mock_features.map((feature, index) => ({"name": feature.feature_name, "checked" : true, index : index}))
            search_query: '',
            search_filtered_list : null
        };
    }

    onNewFeaturesLoaded = (new_features) => {
        const new_formatted_features = new_features.map((feature, index) => ({"name": feature, "checked" : true, index : index}))
        this.setState({formatted_features: new_formatted_features})
        this.setState({search_filtered_list: this.getUpdatedFilteredList(new_formatted_features, this.state.search_query)})
    }

    onChangeSearch = query => {
        if (query.target.value !== '') {
            this.setState({ search_query: query.target.value });
        }

        const updated_filtered_list = this.getUpdatedFilteredList(this.state.formatted_features, query.target.value)
        this.setState({ search_filtered_list: updated_filtered_list })
    };

    getUpdatedFilteredList = (features_list, query) => {
        if (features_list != null) {
            return features_list.filter((feature) => {
                if (query === "") {
                    return features_list
                } else {
                    return feature.name.toLowerCase().includes(query.toLowerCase())
                }
            })
        } else {
            return null
        }

    }

    onChangeCheckbox = i => {
        let formatted_features = [...this.state.formatted_features];  // Make a shallow copy
        let formatted_features_item = {...formatted_features[i]};  // Get the element we want to update
        formatted_features_item.checked = !formatted_features_item.checked  // Change it
        formatted_features[i] = formatted_features_item // Replace it in the array's copy

        const updated_filtered_list = this.getUpdatedFilteredList(formatted_features, this.state.search_query)
        this.setState({formatted_features: formatted_features, search_filtered_list: updated_filtered_list})  // And finally replace the array in the state
    }

    onUncheckButtonClick = () => {
        if (this.state.formatted_features != null) {
            let formatted_features = [...this.state.formatted_features];  // Make a shallow copy
            this.state.search_filtered_list.map((feature) => (  // For each feature currently displayed...
                formatted_features[feature.index].checked = false
            ))

            const updated_filtered_list = this.getUpdatedFilteredList(formatted_features, this.state.search_query)
            this.setState({formatted_features: formatted_features, search_filtered_list: updated_filtered_list})  // And finally replace the array in the state
        }
    }

    onCheckButtonClick = () => {
        if (this.state.formatted_features != null) {
            let formatted_features = [...this.state.formatted_features];  // Make a shallow copy
            this.state.search_filtered_list.map((feature) => (  // For each feature currently displayed...
                formatted_features[feature.index].checked = true
            ))

            const updated_filtered_list = this.getUpdatedFilteredList(formatted_features, this.state.search_query)
            this.setState({formatted_features: formatted_features, search_filtered_list: updated_filtered_list})  // And finally replace the array in the state
        }
    }

    onClearSearchButtonClick = () => {
        const updated_filtered_list = this.getUpdatedFilteredList(this.state.formatted_features, '')
        this.setState({search_filtered_list: updated_filtered_list, search_query: ''})  // And finally replace the array in the state
    }

    render() {
        return (
            <Container style={{height: '100vh'}}>
                <Row style={{height: '100%'}} className="d-flex flex-row justify-content-center align-items-center">

                    <Col className="col-lg-3 col-12 d-flex flex-column" style={{height: "80vh"}}>
                        <Row className="my_row py-2 d-flex flex-row" style={{flexGrow:'1'}}>
                            <ModelSelection />
                        </Row>
                        <Row className="my_row py-1 d-flex flex-row">
                            <AgglomerativeClustering />
                        </Row>
                    </Col>

                    <Col className="col-lg-6 col-12 d-flex flex-column justify-content-center" style={{height: "80vh"}}>
                        <Row className="my_row mx-1 py-2 d-flex flex-row" style={{flexGrow:'1'}}>
                            <DataVisualization/>
                        </Row>
                        <Row className="my_row mx-1 py-2">
                            <DatasetSelector onNewFeaturesLoaded={this.onNewFeaturesLoaded} />
                        </Row>
                    </Col>

                    <Col className="col-lg-3 col-12 d-flex flex-column" style={{height: "80vh"}}>
                        <Row className="my_row py-2" style={{flexGrow:'1'}}>
                            <FeatureSelection search_query={this.state.search_query}
                                              onChangeSearch={this.onChangeSearch}
                                              search_filtered_list={this.state.search_filtered_list}
                                              onCheckButtonClick={this.onCheckButtonClick}
                                              onUncheckButtonClick={this.onUncheckButtonClick}
                                              onClearSearchButtonClick={this.onClearSearchButtonClick}
                                              onChangeCheckbox={this.onChangeCheckbox}/>
                        </Row>
                    </Col>

                </Row>
            </Container>
        )
    }

}

export default FullPage;