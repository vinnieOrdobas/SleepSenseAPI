import pytest
import numpy as np
import json
import joblib
import os

from parameter_handler import ParameterHandler

@pytest.fixture
def setup_parameter_handler():
    # Mock HTML form data
    form_data = {
        'gender': 'Male',
        'age': '36-45',
        'daily_steps': '5000-6500',
        'sleep_duration': '6-7 Hours',
        'height': '175',
        'weight': '70',
        'systolic': '120',
        'diastolic': '80',
        'stress_level': '5',
        'quality_of_sleep': '7',
        'heart_rate': '72'
    }

    # Initialize ParameterHandler
    handler = ParameterHandler(form_data)
    
    return handler

def test_process_inputs(setup_parameter_handler):
    # Instantiate handler from setup
    handler = setup_parameter_handler

    # Scaler path
    scaler_path = os.path.join('models', 'scalers', 'scaler.pkl')

    # Instantiate saved scaler
    scaler = joblib.load(scaler_path)

    # Expected result
    expected_inputs = [
        0,  # Gender (Male)
        1,  # Age (36-45)
        1,  # Daily Steps (5000-6500)
        1,  # Sleep Duration (6-7 Hours)
        0,  # BMI Category (Normal)
        0,  # Blood Pressure (Normal)
        7,  # Quality of Sleep
        5,  # Stress Level
        72  # Heart Rate 
    ]

    # Process inputs
    processed_inputs = handler.process_inputs()

    # Asserts that processed inputs are of correct shape (1 dimension, number of features)
    assert processed_inputs.shape == (1, len(expected_inputs))
    # Asserts that the flattened arrays of (`processed_inputs`) and the scaled (`expected_inputs`) match
    np.testing.assert_array_almost_equal(
        processed_inputs.flatten(),
        scaler.transform([expected_inputs]).flaten()
    )

def test_map_prediction(setup_parameter_handler):
    handler = setup_parameter_handler

    # Test cases for different predicted classes
    assert handler._map_prediction(0) == 'None'
    assert handler._map_prediction(1) == 'Insomnia'
    assert handler._map_prediction(2) == 'Sleep Apnea'
    assert handler._map_prediction(99) == 'Unknown'  # Fallback case

def test_format_output(setup_parameter_handler):
    handler = setup_parameter_handler

    predicitions = np.array([0.1, 0.7, 0.2]) # Predictors Output
    confidence = [0.1, 0.7, 0.2] # Confidence scores

    expected_result = {
        'prediction': 'Insomnia',
        'confidence': {
            'None': 0.1,
            'Insomnia': 0.7,
            'Sleep Apnea': 0.2
        }
    }
    
    formatted_output = handler.format_output(predicitions, confidence)
    assert json.loads(formatted_output) == expected_result