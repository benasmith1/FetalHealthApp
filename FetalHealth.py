#Bena Smith
#Import necessary libraries
import streamlit as st
import pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Suppress warnings
import warnings
warnings.filterwarnings("ignore")

# Set up the title and description of the app
st.title('Fetal Health Classifation: A Machine Learning App') 
st.write("Bena Smith")
st.write("Utilize our advanced machine learning application to predict fetal health classififcation")

# Display an image of penguins
st.image('fetal_health_image.gif', width = 400)

# Load the pre-trained model from the pickle file
dt_pickle = open('dt_fetalhealth_model.pickle', 'rb') 
rf_pickle = open('rf_fetalhealth_model.pickle', 'rb') 
ada_pickle = open('ada_fetalhealth_model.pickle', 'rb') 
vote_pickle = open('vote_fetalhealth_model.pickle', 'rb') 

clf_dt = pickle.load(dt_pickle) 
clf_rf = pickle.load(rf_pickle) 
clf_ada = pickle.load(ada_pickle) 
clf_vote = pickle.load(vote_pickle) 

dt_pickle.close()
rf_pickle.close()
ada_pickle.close()
vote_pickle.close()

fetalhealth_train = pd.read_csv("fetal_health.csv")

# Create a sidebar for input collection
st.sidebar.header('Fetal Health Features Input')

# User file input
file_input = st.sidebar.file_uploader("Upload your data")

# Example dataframe
st.sidebar.warning('Ensure your data strictly follows the format outlined below:', icon="⚠️")
example_upload = pd.read_csv("fetal_health_user.csv")
st.sidebar.write(example_upload.head())

# Select type of model
model_type = st.sidebar.radio(
  "Choose model for prediction",
  options=["Random Forest", "Decision Tree", "AdaBoost", "Soft Voting"]
  )


# Function to change column "Predicted Fetal Health" class numbers to the predicted class
def change_fetalhealth_values(preds):
  i = 0 
  while i < len(preds):
    if preds[i] == 1:
      preds[i] = "Normal"
    if preds[i] == 2:
      preds[i] = "Suspect"
    if preds[i] == 3:
      preds[i] = "Pathological"
    i += 1
  return preds

# Function to get indices of column "Predicted Fetal Health" 
def get_indices(arr, string):
  arr = np.array(arr)
  indices = np.where(arr == string) #https://www.geeksforgeeks.org/numpy-where-in-python/ 
  indices = np.asarray(indices)[0]
  return(indices) 

if not file_input:
  st.info("⏏  Please upload data in sidebar to proceed")

#sidebar file input
if file_input:
  st.success("✅ CSV file uploaded successfully")
  st.sidebar.info(f"You selected: {model_type}")

  fetalhealth_test = pd.read_csv(file_input)
  
  if model_type == "Random Forest":
    #get predictions and prediction probabilities from rf
    preds = clf_rf.predict(fetalhealth_test)
    probas = clf_rf.predict_proba(fetalhealth_test)
    probas = np.max(probas, axis=1) # google gemini ai was used to write this line of code: see appendix

  if model_type == "Decision Tree":
    #get predictions and prediction probabilities from dt
    preds = clf_dt.predict(fetalhealth_test)
    probas = clf_dt.predict_proba(fetalhealth_test)
    probas = np.max(probas, axis=1) # google gemini ai was used to write this line of code: see appendix

  if model_type == "AdaBoost":
    #get predictions and prediction probabilities from ada
    preds = clf_ada.predict(fetalhealth_test)
    probas = clf_ada.predict_proba(fetalhealth_test)
    probas = np.max(probas, axis=1) # google gemini ai was used to write this line of code: see appendix

  if model_type == "Soft Voting":
    #get predictions and prediction probabilities from voting
    preds = clf_vote.predict(fetalhealth_test)
    probas = clf_vote.predict_proba(fetalhealth_test)
    probas = np.max(probas, axis=1) # google gemini ai was used to write this line of code: see appendix

  # change structure of predictions and probabilities
  preds = pd.Series(preds)
  preds = change_fetalhealth_values(preds)
  probas = pd.Series(probas)

  # add predictions and probabilities to df
  fetalhealth_test["Predicted Fetal Health"] = preds
  fetalhealth_test["Predicted Probability"] = probas

  # add color to the df based on predicted fetal health
  # https://medium.com/@romina.elena.mendez/transform-your-pandas-dataframes-styles-colors-and-emojis-bf938d6e98a2 
  fetalhealth_test = fetalhealth_test.style.set_properties(**{'background-color': 'lime', 'color': 'white'}, subset=pd.IndexSlice[get_indices(preds, "Normal"), "Predicted Fetal Health"])\
    .set_properties(**{'background-color': 'yellow', 'color': 'black'},subset=pd.IndexSlice[get_indices(preds, "Suspect"), "Predicted Fetal Health"])\
    .set_properties(**{'background-color': 'orange', 'color': 'white'},subset=pd.IndexSlice[get_indices(preds, "Pathological"), "Predicted Fetal Health"])

  st.write("Scroll to the right of your dataframe below to view predicted fetal health and predicted probabilities")
  st.write(fetalhealth_test)


# #----------------------------------------------------------

if file_input:
  st.write(f'We used a machine learning model ({model_type}) to predict fetal health. '
         'Illustrations of model performance on validation data are displayed below')

  st.subheader("Prediction Performance on Validation Dataset")

  if model_type == "Decision Tree":
    featimp_img = "figures/feature_imp_dt.svg"
    conf_matrix_img = "figures/confusion_mat_dt.svg"
    class_report_img = "figures/class_report_dt.csv"
  if model_type == "Random Forest":
    featimp_img = "figures/feature_imp_rf.svg"
    conf_matrix_img = "figures/confusion_mat_rf.svg"
    class_report_img = "figures/class_report_rf.csv"
  if model_type == "AdaBoost":
    featimp_img = "figures/feature_imp_ada.svg"
    conf_matrix_img = "figures/confusion_mat_ada.svg"
    class_report_img = "figures/class_report_ada.csv"
  if model_type == "Soft Voting":
    featimp_img = "figures/feature_imp_vote.svg"
    conf_matrix_img = "figures/confusion_mat_vote.svg"
    class_report_img = "figures/class_report_vote.csv"

  # if model_type == "Decision Tree": # decsion tree is too big and not really informative so I won't include it on the site
  # # Showing additional items in tabs
  #   tab1, tab2, tab3, tab4 = st.tabs(["Feature Importance", "Confusion Matrix", "Classification Report", "Decision Tree"])
  #else: 
  tab1, tab2, tab3 = st.tabs(["Feature Importance", "Confusion Matrix", "Classification Report"])

  # Tab 1: Feature Importance Visualization
  with tab1:
      st.write("### Feature Importance")
      st.image(featimp_img)
      st.caption("Features used in this prediction are ranked by relative importance.")

  # Tab 2: Confusion Matrix
  with tab2:
      st.write("### Confusion Matrix")
      st.image(conf_matrix_img)
      st.caption("Confusion Matrix of model predictions.")

  # Tab 3: Classification Report
  with tab3:
      st.write("### Classification Report")
      report_df = pd.read_csv(class_report_img, index_col = 0).transpose()
      st.dataframe(report_df.style.background_gradient(cmap='RdBu').format(precision=2))
      st.caption("Classification Report: Precision, Recall, F1-Score, and Support for each species.")

  # if model_type == "Decision Tree":
  #   # Tab 4: Visualizing Decision Tree
  #   with tab4:
  #       st.write("### Decision Tree Visualization")
  #       st.image('dt_visual.svg')
  #       st.caption("Visualization of the Decision Tree used in prediction.")