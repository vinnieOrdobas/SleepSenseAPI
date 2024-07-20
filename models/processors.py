import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

class DataProcessor:
    def __init__(self, df):
        self.df = df

    def process(self):
        # Drop 'Person ID' column
        self.df.drop(columns=['Person ID', 'Occupation'], axis=1, inplace=True)

        # Replace NaN values in 'Sleep Disorder' column
        self.df['Sleep Disorder'] = self.df['Sleep Disorder'].fillna('None')

        # Apply blood pressure categorization
        self.df['Blood Pressure'] = self.df.apply(self.categorize_blood_pressure, axis=1)

        # Transform 'Physical Activity Level' to hours per day and rename column
        self.df['Daily Physical Activity'] = self.df['Physical Activity Level'] / 60
        self.df.drop(columns=['Physical Activity Level'], axis=1, inplace=True)

        # Create bins for 'Age'
        bins_age = [26, 35, 45, 55, 59]
        labels_age = ['26-35', '36-45', '46-55', '55-59']
        self.df['Age'] = self.create_bins('Age', bins_age, labels_age, True)

        # Create bins for 'Daily Steps'
        bins_steps = [4000, 5500, 7000, 8500, 10000]
        labels_steps = ['Under 4000', '5000-6500', '7000-9000', 'Over 10000']
        self.df['Daily Steps'] = self.df['Daily Steps'].clip(lower=4000, upper=10000)
        self.df['Daily Steps'] = self.create_bins('Daily Steps', bins_steps, labels_steps, True)

        # Create bins for 'Sleep Duration'
        bins_sleep = [5, 6, 7, 8, 9]
        labels_sleep = ["5-6 hours", "6-7 hours", "7-8 hours", "8-9 hours"]
        self.df['Sleep Duration'] = self.create_bins('Sleep Duration', bins_sleep, labels_sleep, False)
        
        # Replace 'Normal Weight' with 'Normal' in 'BMI Category'
        self.df['BMI Category'] = self.df['BMI Category'].replace('Normal Weight', 'Normal')

        self.df = self.attribute_mapping()

        return self.df

    def categorize_blood_pressure(self, row):
        blood_pressure = row['Blood Pressure']
        systolic, diastolic = map(int, blood_pressure.split('/'))

        # Check if the blood pressure is within normal range
        if systolic <= 120 and diastolic <= 80:
            return 0

        # Blood pressure is high otherwise
        return 1
    
    def create_bins(self, column, bins, labels, right):
        return pd.cut(self.df[column], bins=bins, labels=labels, right=right, include_lowest=True)

    def attribute_mapping(self):
        mapping_dict = {
            'Gender': {'Male': 0, 'Female': 1},
            'Age': {'26-35': 0, '36-45': 1, '46-55': 2, '55-59': 3},
            'Sleep Duration': {'5-6 hours': 0, '6-7 hours': 1, '7-8 hours': 2, '8-9 hours': 3},
            'BMI Category': {'Normal': 0, 'Overweight': 1, 'Obese': 2},
            'Daily Steps': {'Under 4000': 0, '5000-6500': 1, '7000-9000': 2, 'Over 10000': 3},
            'Sleep Disorder': {'None': 0, 'Insomnia': 1, 'Sleep Apnea': 2}
        }

        for column, mapping in mapping_dict.items():
            self.df[column] = self.df[column].map(mapping)
        
        return self.df

class ModelInputs:
    def __init__(self, df):
        self.df = df
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    def process(self):
        self.df['Sleep Disorder'] = LabelEncoder().fit_transform(self.df['Sleep Disorder'])

        x = self.df.drop(['Sleep Disorder'], axis=1)
        y = self.df['Sleep Disorder']
        # Split data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=.42, random_state=42)
        
        # Standardize features
        scaler = StandardScaler()
        self.X_train = scaler.fit_transform(X_train)
        self.X_test = scaler.transform(X_test)
        self.y_train = y_train
        self.y_test = y_test

        return self