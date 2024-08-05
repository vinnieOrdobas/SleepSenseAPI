import pytest
import numpy as np
import pandas as pd
import json
import joblib
import os
from models.parameter_handler import ParameterHandler

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
        'heart_rate': '72',
        'daily_physical_activity': 'Medium'
    }

    # Initialize ParameterHandler
    handler = ParameterHandler(form_data)
    
    return handler

def test_process_inputs(setup_parameter_handler):
    handler = setup_parameter_handler
    
    # Process the inputs
    processed_inputs = handler.process_inputs()

    # Scaler path
    scaler_path = os.path.join('models', 'scalers', 'scaler.pkl')

    # Load the scaler
    scaler_data = joblib.load(scaler_path)
    scaler = scaler_data['scaler']
    feature_names = scaler_data['feature_names']

    # Expected values before scaling
    expected_values = {
        'Gender': [0],  # Male
        'Age': [1],  # 36-45
        'Sleep Duration': [1],  # 6-7 Hours
        'Quality of Sleep': [7],
        'Stress Level': [5],
        'BMI Category': [0],  # Normal
        'Blood Pressure': [0],  # Normal
        'Heart Rate': [72],
        'Daily Steps': [1],  # 5000-6500
        'Daily Physical Activity': [1]  # Medium
    }

    # Create DataFrame with the correct feature names
    expected_df = pd.DataFrame(expected_values, columns=feature_names)

    # Transform the expected values using the scaler
    scaled_expected_values = scaler.transform(expected_df)

    # Check the shape of processed inputs
    assert processed_inputs.shape == (1, len(feature_names))  # Ensure the number of features matches the scaler

    # Assert that processed inputs are close to the scaled expected values
    assert np.allclose(processed_inputs, scaled_expected_values, rtol=1e-2)

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

    expected_result = {
        'prediction': 'Insomnia',
        'confidence': {
            'None': 0.1,
            'Insomnia': 0.7,
            'Sleep Apnea': 0.2
        }
    }
    
    formatted_output = handler.format_output(predicitions)
    assert json.loads(formatted_output) == expected_result