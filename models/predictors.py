import joblib
import os
import numpy as np
from parameter_handler import ParameterHandler

from sklearn.base import is_classifier

class Predictors:
    def __init__(self, form_data, model_dir='models/predictors'):
        '''
        Initialize models and output prediction
        '''
        self.parameter_handler = ParameterHandler(form_data)
        self.model_dir = model_dir
        self.models = self._load_models()

    def _load_models(self):
        '''
        Load trained models using joblib
        '''

        models = {}
        for filename in os.listdir(self.model_dir):
            if filename.endswith('.pkl'):
                model_name = filename.replace('.pkl', '')
                model_path = os.path.join(self.model_dir, filename)
                models[model_name] = joblib.load(model_path)
        return models
    
    def predict(self, X):
        '''
        Make predictions given new observation, through soft voting
        '''

        model_predictions = []

        for model_name, model in self.models.items():
            if is_classifier(model):
                preds = model.predict_proba(X)
                model_predictions.append(preds)
            
        if not model_predictions:
            raise ValueError('No predictions found')
        
        avg_predictions = np.mean(model_predictions, axis=0)

        return np.argmax(avg_predictions, axis=1)
