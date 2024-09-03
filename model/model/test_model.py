import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm.notebook import tqdm
from joblib import dump, load

mlp_classifier = load('/storage/ice1/1/7/khayashi31/Hacklytics/Model/trained_mlp_model.joblib')

test_data = pd.read_csv('/storage/ice1/1/7/khayashi31/Hacklytics/feature_engineering_data.csv', skiprows=range(1, int(0.8*34085)))
y = test_data['label']
X = test_data.drop(columns=['Unnamed: 0', 'Brief Summary', 'Conditions', 'Interventions', 'Primary Outcome Measures', 'Secondary Outcome Measures',
                'Other Outcome Measures', 'Sponsor', 'Collaborators', 'Sex', 'Age', 'Phases', 'Enrollment', 'Funder Type', 'Study Type', 'Study Design', 'label', 'Allocation', 'Intervention Model', 'Masking', 'Primary Purpose'] )
print(len(test_data))

predicted_probs = pd.DataFrame(mlp_classifier.predict_proba(X))
result = pd.concat([predicted_probs.reset_index(drop=True), y.reset_index(drop=True)], axis=1)
result.to_csv('/storage/ice1/1/7/khayashi31/Hacklytics/predicted_probs.csv')