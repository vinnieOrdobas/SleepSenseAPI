import numpy as np
import joblib
import pandas as pd

class ParameterHandler:
    def __init__(self, form_data, scaler_path='models/scalers/scaler.pkl'):
        self.form_data = form_data
        self.scaler_data = joblib.load(scaler_path)
        self.scaler = self.scaler_data['scaler']
        self.feature_names = self.scaler_data['feature_names']
        self.mapping_dict = {
            'Gender': { 'Male': 0, 'Female': 1},
            'Age': { '26-35': 0, '36-45': 1, '46-55': 2, '55-59': 3 },
            'Sleep Duration': { '5-6 Hours': 0, '6-7 Hours': 1, '7-8 Hours': 2, '8-9 Hours': 3 },
            'BMI Category': { 'Normal': 0, 'Overweight': 1, 'Obese': 2 },
            'Daily Steps': { 'Under 4000': 0, '5000-6500': 1, '7000-9000': 2, 'Over 10000': 3 },
            'Sleep Disorder': { 'None': 0, 'Insomnia': 1, 'Sleep Apnea': 2 },
            'Daily Physical Activity': { 'Low': 0, 'Medium': 1, 'High': 2 }
        }

    def process_inputs(self):
        '''
        Returns numpy array to feed to predictors
        '''
        inputs = self._normalize_inputs()
        inputs_df = pd.DataFrame([inputs], columns=self.feature_names)
        scaled_inputs = self.scaler.transform(inputs_df)
        return np.array(scaled_inputs).reshape(1, -1)
    
    def format_output(self, predictions):
        '''
        Formats prediction to JSON output
        '''

        predicted_class = np.argmax(predictions)
        confidence_score = predictions.flatten().tolist()

        class_labels = ['None', 'Insomnia', 'Sleep Apnea']

        result = {
            'prediction': self._map_prediction(predicted_class),
            'confidence': { label: confidence_score[i] for i, label in enumerate(class_labels) }
        }

        return result

    
    def _normalize_inputs(self):
        '''
        Normalizes form data into suitable inputs
        '''
        gender = self.mapping_dict['Gender'][self.form_data['gender']]
        age = self.mapping_dict['Age'][self.form_data['age']]
        daily_steps = self.mapping_dict['Daily Steps'][self.form_data['daily_steps']]
        sleep_duration = self.mapping_dict['Sleep Duration'][self.form_data['sleep_duration']]
        bmi_category = self._calculate_bmi_category()
        blood_pressure = self._categorize_blood_pressure()
        stress_level = int(self.form_data['stress_level'])
        quality_of_sleep = int(self.form_data['quality_of_sleep'])
        heart_rate = int(self.form_data['heart_rate'])
        daily_physical_activity = self.mapping_dict['Daily Physical Activity'][self.form_data['daily_physical_activity']]
        
        return [
            gender, age, sleep_duration, quality_of_sleep, 
            stress_level, bmi_category, blood_pressure,
            heart_rate, daily_steps, daily_physical_activity
        ]
    
    def _calculate_bmi_category(self):
        height = float(self.form_data['height'])
        weight = float(self.form_data['weight'])

        bmi = weight / (height / 100) ** 2

        if bmi < 25:
            return self.mapping_dict['BMI Category']['Normal']
        elif 25 <= bmi < 30:
            return self.mapping_dict['BMI Category']['Overweight']
        else:
            return self.mapping_dict['BMI Category']['Obese']
    
    def _categorize_blood_pressure(self):
        systolic = float(self.form_data['systolic'])
        diastolic = float(self.form_data['diastolic'])

        # Categorize blood pressure
        if systolic <= 120 and diastolic <= 80:
            return 0  # Normal
        else:
            return 1  # High
    
    def _map_prediction(self, predicted_class):
        reverse_mapping = { v: k for k, v in self.mapping_dict['Sleep Disorder'].items() }

        return reverse_mapping.get(predicted_class, 'Unknown')
