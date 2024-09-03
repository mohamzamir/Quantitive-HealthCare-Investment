
import numpy as np
import pandas as pd

from transformers import pipeline
from sentence_transformers import SentenceTransformer

raw_data = pd.read_csv('/storage/ice1/1/7/khayashi31/Hacklytics/ctg-studies.csv')
print('Successfully Read csv')

keep_columns = ['Brief Summary', 'Conditions', 'Interventions', 'Primary Outcome Measures', 'Secondary Outcome Measures',
                'Other Outcome Measures', 'Sponsor', 'Collaborators', 'Sex', 'Age', 'Phases', 'Enrollment', 'Funder Type', 'Study Type', 'Study Design', ] 

selected_columns_data = raw_data[keep_columns]
with_phase = selected_columns_data[~selected_columns_data['Phases'].isna()] # only keep columns with non null 'Phase' 

with_phase['label'] = np.where(with_phase['Phases'] == 'PHASE4', 1, 0) # Set label (target y) depending on Phase 4 or not


feature_engineering_data = with_phase.copy()
feature_engineering_data['female'] = np.where((feature_engineering_data['Sex']=='FEMALE') | (feature_engineering_data['Sex']=='ALL'), 1, 0)
feature_engineering_data['male'] = np.where((feature_engineering_data['Sex']=='MALE') | (feature_engineering_data['Sex']=='ALL'), 1, 0)

feature_engineering_data = pd.concat([feature_engineering_data, pd.get_dummies(feature_engineering_data['Funder Type'], prefix='Funder_Type')], axis=1) # One-hot encode funder type
feature_engineering_data = pd.concat([feature_engineering_data, feature_engineering_data['Age'].str.get_dummies(sep=', ')], axis=1) # One-hot encode age

study_design_attributes = feature_engineering_data['Study Design'].str.split('|', expand=True) # One-hot encode study design information (multi-variate within string)
for i, attribute in enumerate(['Allocation', 'Intervention Model', 'Masking', 'Primary Purpose']):
    feature_engineering_data[attribute] = study_design_attributes[i].str.split(': ').str[1]
feature_engineering_data = pd.concat([feature_engineering_data, pd.get_dummies(feature_engineering_data[['Allocation', 'Intervention Model', 'Masking', 'Primary Purpose']])], axis=1)


feature_engineering_data = pd.concat([feature_engineering_data, pd.get_dummies(feature_engineering_data['Sponsor'], prefix='Sponsor')], axis=1)
feature_engineering_data = pd.concat([feature_engineering_data, pd.get_dummies(feature_engineering_data['Collaborators'], prefix='Collaborators')], axis=1)

print('Starting Transformer Work')

feature_engineering_data[['Brief Summary', 'Conditions', 'Interventions', 'Primary Outcome Measures', 'Secondary Outcome Measures', 'Other Outcome Measures']] = feature_engineering_data[['Brief Summary', 'Conditions', 'Interventions', 'Primary Outcome Measures', 'Secondary Outcome Measures', 'Other Outcome Measures']].fillna('Missing Data')

# Use Sentence Transformer 
sentence_transformer = SentenceTransformer('paraphrase-MiniLM-L6-v2')
print('Starting Brief Summary')
brief_summary_vector = sentence_transformer.encode(feature_engineering_data['Brief Summary'].values.tolist())
feature_engineering_data = pd.concat([feature_engineering_data.reset_index(drop=True), pd.DataFrame(brief_summary_vector).add_prefix('brief_summary_vector_')], axis=1) 

print('Starting Summary')
conditions_vector = sentence_transformer.encode(feature_engineering_data['Conditions'].values.tolist())
feature_engineering_data = pd.concat([feature_engineering_data.reset_index(drop=True), pd.DataFrame(conditions_vector).add_prefix('conditions_vector_')], axis=1) 

print('Starting Interventions')
interventions_vector = sentence_transformer.encode(feature_engineering_data['Interventions'].values.tolist())
feature_engineering_data = pd.concat([feature_engineering_data.reset_index(drop=True), pd.DataFrame(interventions_vector).add_prefix('interventions_vector_')], axis=1) 

print('Starting Primary Outcome Measures')
primary_outcome_measures_vector = sentence_transformer.encode(feature_engineering_data['Primary Outcome Measures'].values.tolist())
feature_engineering_data = pd.concat([feature_engineering_data.reset_index(drop=True), pd.DataFrame(primary_outcome_measures_vector).add_prefix('primary_outcome_measures_vector_')], axis=1) 

print('Starting Secondary Outcome Measures')
secondary_outcome_measures_vector = sentence_transformer.encode(feature_engineering_data['Secondary Outcome Measures'].values.tolist())
feature_engineering_data = pd.concat([feature_engineering_data.reset_index(drop=True), pd.DataFrame(secondary_outcome_measures_vector).add_prefix('secondary_outcome_measures_vector_')], axis=1) 

print('Starting Other Outcome Measures')
other_outcome_measures_vector = sentence_transformer.encode(feature_engineering_data['Other Outcome Measures'].values.tolist())
feature_engineering_data = pd.concat([feature_engineering_data.reset_index(drop=True), pd.DataFrame(other_outcome_measures_vector).add_prefix('other_outcome_measures_vector_')], axis=1) 

print('Converting Dataset to csv')
feature_engineering_data.to_csv('/storage/ice1/1/7/khayashi31/Hacklytics/feature_engineering_data.csv')

