# Preliminaries
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from copy import deepcopy

# File I/O
import lzma
import pickle

# Model
from xgboost import XGBClassifier

# GUI
import streamlit as st
st.set_option('deprecation.showPyplotGlobalUse', False)


# Caching methods for memory management

@st.cache(allow_output_mutation=True)
def load_model(file_path):
    
    # Load Models
    model = XGBClassifier()
    model.load_model(file_path)
    return model

@st.cache(allow_output_mutation=True)
def load_xz(file_path):
    # Load Explainers
    with lzma.open(file_path, 'rb') as f:
        xz_object = pickle.load(f)
    return xz_object

@st.cache(allow_output_mutation=False)
def load_xz_non_mutable(file_path):
    # Load Explainers
    with lzma.open(file_path, 'rb') as f:
        xz_object = pickle.load(f)
    return xz_object


#### START OF APP ####
st.title('Resignation Model V2 & V3 With SHAP Explanation')

# Categorical options
job_classes = sorted(['Finance', 'Rank & File', 'BPO', 'Analyst', 'C-suite',
       'Information Technology', 'Managerial', 'Specialist', 'Supervisor',
       'Director', 'Team Leader', 'Head', 'Education', 'Human Resource',
       'Unclassified Low Income', 'Minimum Wage Earners',
       'Experienced Pay - Unclassified', 'Assistant', 'Senior Management'])

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

v2_binary_dict = {
    0: 'Resigned',
    1: 'Not Resigned',
           }


# Create item for prediction
def process_data(df):
    """
    Get dummies for categorical columns and apply log transform to
    numerical features.
    """
    df = pd.get_dummies(df,columns=['Gender', 'Job Class',
                                    'Employment Status'])
    
    X = pd.DataFrame({0: {feature:
                          df.loc[0, feature] if(feature in df.columns)
                          else 0 for feature in features_list}}).transpose()
    X_log = X.copy()
    X_log[log_cols] = X_log[log_cols].apply(lambda x: np.log(x+1))
    return X, X_log

def local_shap_beeswarm(shap_values_local, shap_values_global):
    """Creates local shap values to align with a global one in plotting."""
    shap_values_temp = deepcopy(shap_values_global)
    shap_values_temp.values = np.array(
        [shap_values_global.values.min(axis=0),
         shap_values_global.values.max(axis=0),
         shap_values_local.values[0]])
    shap_values_temp.data = np.array(
        [shap_values_global.data.min(axis=0),
         shap_values_global.data.max(axis=0),
         shap_values_local.data[0]])
    shap_values_temp.base_values = np.array(
        np.repeat(shap_values_global.base_values.min(axis=0), 3))
    return shap_values_temp

def plot_beeswarm(shap_values_local, shap_values_global):
    """Create SHAP beeswarm plot with local point highlighted."""
    shap.plots.beeswarm(
        shap_values_global, show=False, max_display=30, alpha=0.1,
        color=plt.get_cmap('Wistia'),
        color_bar_label='Global Feature Values')

    shap.plots.beeswarm(
        local_shap_beeswarm(shap_values_local, shap_values_global),
        show=False, max_display=30,
        alpha=[0, 0, 1], color_bar_label='Local Feature Values')

    st.pyplot(bbox_inches='tight', dpi=300, pad_inches=0)


def make_predictions(with_shap=0):
    """
    Make predictions using models and show output.

    Parameters
    ----------
    with_shap : int
        0 for no SHAP Plots, 1 for Local SHAP plots, 2 for Local & Global SHAP Plots
    """

    global v2_data, v3_data
    global prediction_v2_1, prediction_v2_2, prediction_v2, prediction_v3
    global confidence_v2_1, confidence_v2_2, confidence_v3

    v2_data, v3_data = process_data(df)
    v2_data = v2_data[features_list]
    v3_data = v3_data[features_list]
    
    # Run Models
    prediction_v2_1 = model_v2_1.predict(v2_data)[0]
    prediction_v2_2 = model_v2_2.predict(v2_data)[0]
    prediction_v2 = (1 - prediction_v2_1) * prediction_v2_2
    confidence_v2_1 = model_v2_1.predict_proba(v2_data)[0].max()
    confidence_v2_2 = model_v2_2.predict_proba(v2_data)[0].max()
    prediction_v3 = model_v3.predict(v3_data)[0]
    confidence_v3 = model_v3.predict_proba(v3_data)[0].max()
    
    # Prettify feature display
    feature_names = [name.replace('_', '\n') for name in features_list]
    
    # Calculate local SHAP values
    shap_values_local_1 = explainer_v2_1(v2_data)
    shap_values_local_2 = explainer_v2_2(v2_data)
    shap_values_local_3 = explainer_v3(v3_data)
    
    # V2 Output
    st.success('V2 Model:' +
               '  \n--------------' +
               '  \nPrediction: ' + v2_predict_dict[prediction_v2] +
               '  \nBinary Probability: ' + str(round(confidence_v2_1*100, 1)) +
               '%' + ('' if prediction_v2_1 else '  \nBinned Probability: ' +
                      str(round(confidence_v2_2*100, 1)) + '%'))
      
    if not(prediction_v2_1):
        st.markdown('**V2 Probabilities**')   
        st.dataframe(pd.DataFrame(
            model_v2_2.predict_proba(v2_data),
            columns=list(v2_predict_dict.values())[1:],
            index=['Probabilities']
            ).applymap(
                lambda x: str(round(100*x, 2)) +'%')
            )
                
    # V2 SHAP
    if with_shap:
        st.markdown('**Local SHAP Plot**')
        if prediction_v2_1 == 0:
            shap.force_plot(
                base_value=explainer_v2_2.expected_value[
                    prediction_v2_2 - 1],
                shap_values=shap_values_local_2.values[
                    0, :, prediction_v2_2 - 1],
                feature_names=feature_names,
                out_names=v2_predict_dict[prediction_v2_2],
                figsize=(22, 4),
                matplotlib=True
            )
            st.pyplot(bbox_inches='tight', dpi=300, pad_inches=0)
            st.caption(":red[Pink bars] push the prediction towards the _" + 
                       v2_predict_dict[prediction_v2_2] + "_ prediction. :blue[Blue bars] drag the "
                       "prediction away from the _" + v2_predict_dict[prediction_v2_2]
                       + "_ assignment."
                       )

            if with_shap==2:
                st.markdown('**Global SHAP Plot for ' + v2_predict_dict[prediction_v2_2] + '**')
                plot_beeswarm(
                    shap_values_local_2[:, :, prediction_v2_2 - 1],
                    shap_values_global_2[:, :, prediction_v2_2 - 1])

        else:
            shap.force_plot(
                base_value=explainer_v2_1.expected_value,
                shap_values=shap_values_local_1.values[0],
                feature_names=feature_names,
                out_names=v2_binary_dict[prediction_v2_1],
                figsize=(22, 4),
                matplotlib=True
            )
            st.pyplot(bbox_inches='tight', dpi=300, pad_inches=0)
            st.caption(":red[Pink bars] push the prediction towards " 
                       "_Not Resigned_. :blue[Blue bars] push the prediction"
                       " towards _Resigned_."
                       )

            if with_shap==2:
                st.markdown('**Global SHAP Plot for Not Resigned**')
                plot_beeswarm(shap_values_local_1, shap_values_global_1)

    # V3 Output
    st.success('V3 Model:' +
               '  \n--------------'
               '  \nPrediction: ' + v3_predict_dict[prediction_v3] +
               '  \nProbability: ' + str(round(confidence_v3*100, 1)) + '%')

    st.markdown('**V3 Probabilities**') 
    st.dataframe(pd.DataFrame(model_v3.predict_proba(v3_data),
                              columns=v3_predict_dict.values(),
                              index=['Probabilities']
                              ).applymap(lambda x: str(round(100*x, 2)) +'%')
                 )
    
    # V3 SHAP
    if with_shap:
        st.markdown('**Local SHAP Plot**')
        shap.force_plot(
            base_value=explainer_v3.expected_value[prediction_v3],
            shap_values=shap_values_local_3.values[0, :, prediction_v3],
            feature_names=feature_names,
            out_names=v3_predict_dict[prediction_v3],
            figsize=(22, 4),
            matplotlib=True
        )
        st.pyplot(bbox_inches='tight', dpi=300, pad_inches=0)
        st.caption(":red[Pink bars] push the prediction towards the _" + 
                   v3_predict_dict[prediction_v3] + "_ prediction. :blue[Blue bars] drag the "
                   "prediction away from the _" + v3_predict_dict[prediction_v3]
                   + "_ assignment."
                   )

        if with_shap==2:
            st.markdown('**Global SHAP Plot for ' + v3_predict_dict[prediction_v3] + '**')
            plot_beeswarm(
                shap_values_local_3[:, :, prediction_v3],
                shap_values_global_3[:, :, prediction_v3])

# Load Models
model_v2_1 = load_model("xgb_model_v2_1.json")
model_v2_2 = load_model("xgb_model_v2_2.json")
model_v3 = load_model("v3_model.json")
    
# Load Explainers
explainer_v2_1 = load_xz('explainer_v2_1.xz')
explainer_v2_2 = load_xz('explainer_v2_2.xz')
explainer_v3 = load_xz('explainer_v3.xz')

# Load Global SHAP VAlues
shap_values_global_1 = load_xz_non_mutable('shap_values_global_1.xz')
shap_values_global_2 = load_xz_non_mutable('shap_values_global_2.xz')
shap_values_global_3 = load_xz_non_mutable('shap_values_global_3.xz')

# Input Data Storage
df = pd.DataFrame()

# Start of input form
with st.form("my_form"):
    c1, c2 = st.columns(2)
    
    with c1:
        df.loc[0, 'Gender'] = st.selectbox(label='Gender',
                                           options=['Male', 'Female'],
                                           index=1
                                           )
        df.loc[0, 'Job Class'] = st.selectbox(label='Job Class',
                                              options=job_classes)
        df.loc[0, 'Employment Status'] = st.selectbox(label='Employment Status',
                                                      options=employment_stats
                                                      )
        
    with c2:
        df.loc[0, 'Salary'] = st.number_input(label='Salary',
                                              min_value=0.00,
                                              value=25000.00,
                                              step=1e-2)
        df.loc[0, 'Tenure (years)'] = st.number_input(label='Tenure',
                                              min_value=0.00,
                                              max_value=100.00,
                                              value=5.00, step=1e-2)
        df.loc[0, 'Age'] = st.number_input(label='Age',
                                           min_value=0,
                                           max_value=100,
                                           value=30)
    
    submitted0 = st.form_submit_button('Predict')
    submitted1 = st.form_submit_button('Predict & Show SHAP Plots (Local)')
    submitted2 = st.form_submit_button('Predict & Show SHAP Plots (Local & Global)')
    
    if submitted0: # Predict Only
        make_predictions(with_shap=0)
        
    if submitted1: # Predict & Plot SHAP (Local)
        make_predictions(with_shap=1)

    if submitted2: # Predict & Plot SHAP (Local & Global)
        make_predictions(with_shap=2)