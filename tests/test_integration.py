import pytest
import json
from app import app
import pdb

@pytest.fixture
def client():
    with app.test_client() as client:
        yield client
    
def test_predict(client):
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

    response = client.post('/predict', data=form_data)
    response_json = json.loads(response.data)

    assert response.status_code == 200
    assert response_json['prediction'] in ['None', 'Insomnia', 'Sleep Apnea']
    assert len(response_json['confidence']) == 3

