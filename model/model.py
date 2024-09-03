
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm.notebook import tqdm
from joblib import dump

chunksize = 3000
from sklearn.neural_network import MLPClassifier
mlp_classifier = MLPClassifier(
    activation='relu',
    solver='adam',
    alpha=0.0001,
    batch_size='auto',
    learning_rate='adaptive',
    shuffle=True,
    verbose=True,
)

validation_set = None
i = 0
for data in tqdm(pd.read_csv('/storage/ice1/1/7/khayashi31/Hacklytics/feature_engineering_data.csv', chunksize=chunksize, iterator=True)):
    if i / 34085 > 0.8:
        break
    y = data['label']
    X = data.drop(columns=['Unnamed: 0', 'Brief Summary', 'Conditions', 'Interventions', 'Primary Outcome Measures', 'Secondary Outcome Measures',
                'Other Outcome Measures', 'Sponsor', 'Collaborators', 'Sex', 'Age', 'Phases', 'Enrollment', 'Funder Type', 'Study Type', 'Study Design', 'label', 'Allocation', 'Intervention Model', 'Masking', 'Primary Purpose'] )
    if validation_set == None: 
        validation_set = (X, y)
        continue
    print('Fitting Data')
    mlp_classifier.partial_fit(X, y, [0, 1])
    predicted_probs = mlp_classifier.predict_proba(validation_set[0])
    print('Validation Score')
    print(mlp_classifier.score(validation_set[0], validation_set[1]))
    i += 1
 
dump(mlp_classifier, 'trained_mlp_model.joblib') 

# print('Pulling Data')
# data = pd.read_csv('/storage/ice1/1/7/khayashi31/Hacklytics/feature_engineering_data.csv', nrows=1000)
# print('Successfully Pulled Data')

# y = data['label']
# X = data.drop(columns=['Unnamed: 0', 'Brief Summary', 'Conditions', 'Interventions', 'Primary Outcome Measures', 'Secondary Outcome Measures',
#                 'Other Outcome Measures', 'Sponsor', 'Collaborators', 'Sex', 'Age', 'Phases', 'Enrollment', 'Funder Type', 'Study Type', 'Study Design', 'label', 'Allocation', 'Intervention Model', 'Masking', 'Primary Purpose'] )

# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.1, random_state=42
# )


# print('Fitting Data')
# mlp_classifier.partial_fit(X_train, y_train, [0, 1])
# print(mlp_classifier.predict(X_test))

# print('-' * 100)

# print(mlp_classifier.predict_proba(X_test))

# print('-' * 100)

# print(mlp_classifier.score(X_test, y_test))