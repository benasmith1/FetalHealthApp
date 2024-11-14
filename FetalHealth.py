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

file_input = st.sidebar.file_uploader("Upload your data")

model_type = st.sidebar.radio(
  "Choose model for prediction",
  options=["Random Forest", "Decision Tree", "AdaBoost", "Soft Voting"]
  )


#function to change column "Predicted Fetal Health" class numbers to the predicted class
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
  indices = np.where(arr == string)
  indices = np.asarray(indices)[0]
  print(indices)
  return(indices) #[0,1,2,3,4])

  return indices


    


#sidebar file input
if file_input:
    
  fetalhealth_test = pd.read_csv(file_input)
  
  if model_type == "Random Forest":
    #get predictions and prediction probabilities
    preds = clf_rf.predict(fetalhealth_test)
    probas = clf_rf.predict_proba(fetalhealth_test)
    probas = np.max(probas, axis=1)
    print(probas)


    preds = pd.Series(preds)
    preds = change_fetalhealth_values(preds)


    probas = pd.Series(probas)

    fetalhealth_test["Predicted Fetal Health"] = preds
    fetalhealth_test["Predicted Probability"] = probas

    fetalhealth_test = fetalhealth_test.style.set_properties(**{'background-color': 'lime', 'color': 'white'}, subset=pd.IndexSlice[get_indices(preds, "Normal"), "Predicted Fetal Health"])\
      .set_properties(**{'background-color': 'yellow', 'color': 'black'},subset=pd.IndexSlice[get_indices(preds, "Suspect"), "Predicted Fetal Health"])\
      .set_properties(**{'background-color': 'red', 'color': 'white'},subset=pd.IndexSlice[get_indices(preds, "Pathological"), "Predicted Fetal Health"])



    #fetalhealth_test = fetalhealth_test.style.set_properties({"Predicted Fetal Health": "green",})
    #fetalhealth_test=fetalhealth_test.style.set_properties(**{'background-color': 'pink', 'color': 'purple'})

    #fetalhealth_test['Predicted Probability'].style.bar()


    st.write(fetalhealth_test)



#   # Showing Feature Importance plot
# st.write('We used a machine learning model (Decision Tree) to predict the species. '
#         'The features used in this prediction are ranked by relative importance below.')
# st.image('feature_imp.svg')

# #----------------------------------------------------------
# # Showing additional items in tabs
# st.subheader("Prediction Performance")
# tab1, tab2, tab3, tab4 = st.tabs(["Decision Tree", "Feature Importance", "Confusion Matrix", "Classification Report"])

# # Tab 1: Visualizing Decision Tree
# with tab1:
#     st.write("### Decision Tree Visualization")
#     st.image('dt_visual.svg')
#     st.caption("Visualization of the Decision Tree used in prediction.")

# # Tab 2: Feature Importance Visualization
# with tab2:
#     st.write("### Feature Importance")
#     st.image('feature_imp.svg')
#     st.caption("Features used in this prediction are ranked by relative importance.")

# # Tab 3: Confusion Matrix
# with tab3:
#     st.write("### Confusion Matrix")
#     st.image('confusion_mat.svg')
#     st.caption("Confusion Matrix of model predictions.")

# # Tab 4: Classification Report
# with tab4:
#     st.write("### Classification Report")
#     report_df = pd.read_csv('class_report.csv', index_col = 0).transpose()
#     st.dataframe(report_df.style.background_gradient(cmap='RdBu').format(precision=2))
#     st.caption("Classification Report: Precision, Recall, F1-Score, and Support for each species.")

