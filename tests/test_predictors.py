import pytest
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from predictors import Predictors
from parameter_handler import ParameterHandler

@pytest.fixture
def mock_scaler(mocker):
    scaler = mocker.Mock()
    scaler.transform.return_value = np.array([[0, 1, 2, 3]])
    return scaler

@pytest.fixture
def stub_mlr_model(mocker):
    model = LogisticRegression()
    mocker.patch.object(model, 'predict_proba', return_value=np.array([[0.2, 0.5, 0.3]]))
    return model

@pytest.fixture
def stub_rf_model(mocker):
    model = RandomForestClassifier()
    mocker.patch.object(model, 'predict_proba', return_value=np.array([[0.1, 0.6, 0.3]]))
    return model

@pytest.fixture
def stub_gbc_model(mocker):
    model = GradientBoostingClassifier()
    mocker.patch.object(model, 'predict_proba', return_value=np.array([[0.3, 0.4, 0.3]]))
    return model

@pytest.fixture
def mock_load_models(mocker, stub_mlr_model, stub_rf_model, stub_gbc_model, mock_scaler):
    load = mocker.patch('joblib.load')
    feature_names = [
        'Gender',
        'Age',
        'Sleep Duration', 
        'Quality of Sleep', 
        'Stress Level', 
        'BMI Category', 
        'Blood Pressure',
        'Heart Rate', 
        'Daily Steps', 
        'Daily Phyisical Activity'
    ]

    load.side_effect = lambda path: {
        'models/predictors/MLR_model.pkl': stub_mlr_model,
        'models/predictors/RF_model.pkl': stub_rf_model,
        'models/predictors/GBC_model.pkl': stub_gbc_model,
        'models/scalers/scaler.pkl': {'scaler': mock_scaler, 'feature_names': feature_names }
    }[path]
    return load

@pytest.fixture
def predictor_instance(mock_load_models):
    return Predictors(form_data={})

@pytest.fixture
def sample_data():
    return np.array([[5.1, 3.5, 1.4, 0.2]])

def test_load_models(predictor_instance, stub_mlr_model, stub_rf_model, stub_gbc_model):
    expected_models = {
        'MLR_model': stub_mlr_model,
        'RF_model': stub_rf_model,
        'GBC_model': stub_gbc_model
    }

    for model_name, model in expected_models.items():
        assert isinstance(predictor_instance.models[model_name], type(model))

def test_predict(predictor_instance, sample_data):
    prediction = predictor_instance.predict(sample_data)

    expected_proba = np.array([[0.2, 0.5, 0.3], [0.1, 0.6, 0.3], [0.3, 0.4, 0.3]])
    expected_avg_proba = np.mean(expected_proba, axis=0)
    expected_class = np.argmax(expected_avg_proba)
    
    assert np.array_equal(prediction, [expected_class]), f"Expected {[expected_class]} but got {prediction}"
