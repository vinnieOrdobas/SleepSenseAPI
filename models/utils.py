import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

from sklearn.model_selection import learning_curve, cross_val_score, GridSearchCV
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

def plot_accuracy(model, name, X_train, y_train):
    '''
    Plots learning curve for model, test set and training set
    '''
    train_sizes, train_scores, test_scores = learning_curve(
    model, X_train, y_train, cv=5, scoring='accuracy', 
    train_sizes=np.linspace(0.1, 1.0, 10)
    )
    
    train_scores_mean = train_scores.mean(axis=1)
    test_scores_mean = test_scores.mean(axis=1)
    
    plt.plot(train_sizes, train_scores_mean, label='Training score')
    plt.plot(train_sizes, test_scores_mean, label='Test score')
    plt.xlabel('Training set size')
    plt.ylabel('Accuracy')
    plt.title(f"{name}'s Learning Curve")
    plt.legend()
    return plt.show()

# Create function to fit data and score models
def fit_and_score(models, X_train, X_test, y_train, y_test):
  '''
  Fits and evaluates given machine learning models.
  models: dictionary with different ML models
  X_train: training data (no labels)
  X_test: test data (no labels)
  y_train: training labels
  y_test: test labels
  '''
  model_scores = {}
  for name, model in models.items():
    model.fit(X_train, y_train)
    y_preds = model.predict(X_test)
    model_scores[f"{name} Test"] = model.score(X_test, y_test)
    model_scores[f"{name} Training"] = model.score(X_train, y_train)
    model_scores[f"{name} Accuracy Score"] = accuracy_score(y_test, y_preds)
    model_scores[f"{name} Learning Curve"] = plot_accuracy(model, name, X_train, y_train)
  return model_scores

# Print Scores
def print_scores(model_scores):
   '''
   Print scores given hash
   '''
   for k, v in model_scores.items():
      print(f"{k}:{v}\n")

# Confusion Matrix
def plot_confusion_matrix(y_test, y_preds):
  '''
  Plot confusion matrix given labels and predictions
  '''
  fig, ax = plt.subplots(figsize=(3,3))
  labels = ['None', 'Insomnia', 'Sleep Apnea']
  ax = sns.heatmap(confusion_matrix(y_test, y_preds),
                   annot=True,
                   cbar=False,
                   yticklabels=labels,
                   xticklabels=labels
                  )
  plt.xlabel('Predicted label')
  plt.ylabel('True label')

# Cross Validated Metrics
def cross_validated_metrics(model, x, y, cv=5):
   '''
   Returns data frame with cross validated metric results
   '''
   scores = {}
   scores['accuracy'] = np.mean(cross_val_score(model,x, y, cv=cv, scoring='accuracy'))
   scores['precision'] = np.mean(cross_val_score(model, x, y, cv=cv, scoring='precision_weighted'))
   scores['recall'] = np.mean(cross_val_score(model, x, y, cv=cv, scoring='recall_weighted'))
   scores['f1'] = np.mean(cross_val_score(model, x, y, cv=cv, scoring='f1_weighted'))

   return pd.DataFrame(scores, index=[0])

def plot_validated_metrics(cv_metrics):
   '''
   Plots a bar chart with cross validated metrics
   '''
   return cv_metrics.T.plot(kind='bar', title='Cross validated metrics', legend=False)

def fine_tune_model(model, param_grid, X_train, y_train):
   '''
   Returns best model given a parameter grid
   '''
   grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, n_jobs=1, verbose=True)
   grid_search.fit(X_train, y_train)

   return grid_search.best_estimator_

def build_pipeline(model):
   '''
   Builds pipeline given model
   '''
   pipeline = Pipeline([
      ('imputer', SimpleImputer(strategy='mean')),
      ('scaler', StandardScaler()),
      ('classifier', model)
   ])

   return pipeline


def export_models(models):
    '''
    Export models to '../models/predictors'
    '''
    # Define the directory where models will be saved
    model_dir = '../models/predictors'
    
    # Ensure directory exists
    os.makedirs(model_dir, exist_ok=True)

    for model_name, model_instance in models.items():
        # Sanitize model_name for file naming
        sanitized_model_name = model_name.replace(' ', '_')
        filename = os.path.join(model_dir, f"{sanitized_model_name}_model.pkl")
        
        # Save the model
        joblib.dump(model_instance, filename)
        print(f"Model '{model_name}' saved to '{filename}'")

def export_scaler(scaler):
    '''
    Export scaler to process form data
    '''
    # Define the directory and file path where the scaler will be saved
    scaler_dir = '../models/scalers'
    scaler_file = 'scaler.pkl'
    scaler_path = os.path.join(scaler_dir, scaler_file)

    # Ensure the directory exists
    os.makedirs(scaler_dir, exist_ok=True)

    # Save the scaler
    joblib.dump(scaler, scaler_path)