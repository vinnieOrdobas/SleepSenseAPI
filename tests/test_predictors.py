import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from predictors import Predictors

@pytest.fixture
def stub_mlr_model():
    model = LogisticRegression()
    model.predict_proba = MagicMock(return_value= np.array([[0.2, 0.5, 0.3]]))
    return model

@pytest.fixture
def stub_rf_model():
    model = RandomForestClassifier()
    model.predict_proba = MagicMock(return_value=np.array([[0.1, 0.6, 0.3]]))
    return model

@pytest.fixture
def stub_gbc_model():
    model = GradientBoostingClassifier()
    model.predict_proba = MagicMock(return_value=np.array([[0.3, 0.4, 0.3]]))
    return model

@pytest.fixture
def sample_data():
    """
    Fixture to provide sample input data for the tests.
    """
    return np.array([[5.1, 3.5, 1.4, 0.2]])

@pytest.fixture
def mock_load_models(stub_mlr_model, stub_rf_model, stub_gbc_model):
    with patch('os.listdir') as mock_listdir, patch('joblib.load') as mock_load:
        mock_listdir.return_value = ['mlr_model.pkl', 'rf_model.pkl', 'gbc_model.pkl']
        mock_load.side_effect = [stub_mlr_model, stub_rf_model, stub_gbc_model]
        yield mock_load

@pytest.fixture
def predictor_instance(mock_load_models):
    # Create an instance of Predictors
    return Predictors()

def test_load_models(predictor_instance, stub_mlr_model, stub_rf_model, stub_gbc_model):

    expected_models = {
        'mlr_model': stub_mlr_model,
        'rf_model': stub_rf_model,
        'gbc_model': stub_gbc_model
    }

    assert predictor_instance.models == expected_models

def test_predict(predictor_instance, sample_data):
    
    prediction = predictor_instance.predict(sample_data)

    expected_proba = np.array([[0.2, 0.5, 0.3], [0.1, 0.6, 0.3], [0.3, 0.4, 0.3]])
    expected_avg_proba = np.mean(expected_proba, axis=0)
    expected_class = np.argmax(expected_avg_proba)
    assert np.array_equal(prediction, [expected_class]), f"Expected {[expected_class]} but got {prediction}"