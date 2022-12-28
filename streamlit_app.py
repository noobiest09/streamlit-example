# Preliminaries
import numpy as np
import pandas as pd

# Model
from xgboost import XGBClassifier

# GUI
import streamlit as st

"""
Resignation Model V2 & V3 With SHAP Explanation
"""

# categorical options
job_classes = ['Finance', 'Rank & File', 'BPO', 'Analyst', 'C-suite',
       'Information Technology', 'Managerial', 'Specialist', 'Supervisor',
       'Director', 'Team Leader', 'Head', 'Education', 'Human Resource',
       'Unclassified Low Income', 'Minimum Wage Earners',
       'Experienced Pay - Unclassified', 'Assistant', 'Senior Management']

employment_stats = ['Regular', 'Fixed Term', 'Probationary', 'Project-Based']

features_list = ['Gender_Female',
                 'Gender_Male',
                 'Job Class_Analyst',
                 'Job Class_Assistant',
                 'Job Class_BPO',
                 'Job Class_C-suite',
                 'Job Class_Director',
                 'Job Class_Education',
                 'Job Class_Experienced Pay - Unclassified',
                 'Job Class_Finance',
                 'Job Class_Head',
                 'Job Class_Human Resource',
                 'Job Class_Information Technology',
                 'Job Class_Managerial',
                 'Job Class_Minimum Wage Earners',
                 'Job Class_Rank & File',
                 'Job Class_Senior Management',
                 'Job Class_Specialist',
                 'Job Class_Supervisor',
                 'Job Class_Team Leader',
                 'Job Class_Unclassified Low Income',
                 'Employment Status_Fixed Term',
                 'Employment Status_Probationary',
                 'Employment Status_Project-Based',
                 'Employment Status_Regular',
                 'Salary',
                 'Tenure (years)',
                 'Age']

log_cols = ['Salary', 'Tenure (years)', 'Age']

v3_predict_dict = {0: '< 3 Months',
                   1: '3 - 6 Months',
                   2: '6 - 9 Months',
                   3: '9 - 12 Months',
                   4: '> 12 Months'
                   }

v2_predict_dict = {0: 'Passed',
                   1: '< 3 Months',
                   2: '3 - 6 Months',
                   3: '> 6 Months'
                   }

# Input Data
df = pd.DataFrame()

c1, c2 = st.columns(2)

with c1:
    df.loc[0, 'Gender'] = st.selectbox(label='Gender',
                                       options=['Male', 'Female'])
    df.loc[0, 'Employment Status'] = st.selectbox(label='Employment Status',
                                           options=employment_stats)
    df.loc[0, 'Job Class'] = st.selectbox(label='Job Class',
                                          options=job_classes)
with c2:
    df.loc[0, 'Age'] = st.number_input(label='Age', min_value=0, max_value=100,
                                       value=30)
    df.loc[0, 'Tenure'] = st.number_input(label='Tenure', min_value=0,
                                   max_value=100, value=5)
    df.loc[0, 'Salary'] = st.number_input(label='Salary', min_value=0,
                                          max_value=100000, value=25000)

# Create item for prediction
def process_data(df):
    df = pd.get_dummies(df,columns=['Gender', 'Job Class',
                                    'Employment Status'])
    
    X = pd.DataFrame({0: {feature:
                          df.loc[0, feature] if(feature in df.columns)
                          else 0 for feature in features_list}}).transpose()
    X_log = X.copy()
    X_log[log_cols] = X_log[log_cols].apply(lambda x: np.log(x+1))
    return X, X_log


# Load Models
model_v3 = XGBClassifier()
model_v2_1 = XGBClassifier()
model_v2_2 = XGBClassifier()
model_v3.load_model("v3_model.json")
model_v2_1.load_model("xgb_model_v2_1.json")
model_v2_2.load_model("xgb_model_v2_2.json")

# Button
if st.button('Predict'):
    v2_data, v3_data = process_data(df)
    
    # V3 Output
    prediction_v3 = model_v3.predict(v3_data)[0]
    confidence_v3 = model_v3.predict_proba(v3_data)[0].max()
    st.success('V3 Model:' +
               '  \n--------------'
               '  \nPrediction: ' + v3_predict_dict[prediction_v3] +
               '  \nConfidence: ' + str(round(confidence_v3*100, 1)) + '%')
    
    # V2 Output
    prediction_v2_1 = model_v2_1.predict(v2_data)[0]
    prediction_v2_2 = model_v2_2.predict(v2_data)[0]
    prediction_v2 = (1 - prediction_v2_1) * prediction_v2_2
    confidence_v2_1 = model_v2_1.predict_proba(v2_data)[0].max()
    confidence_v2_2 = model_v2_2.predict_proba(v2_data)[0].max()
    st.success('V2: Model' +
               '  \n--------------' +
               '  \nPrediction: ' + v2_predict_dict[prediction_v2] +
               '  \nBinary Confidence :' + str(round(confidence_v2_1*100, 1)) +
               '%' + ('' if prediction_v2_1 else '  \nBinned Probability: ' +
                      str(round(confidence_v2_2*100, 1)) + '%'))
