#!/usr/bin/env python
# coding: utf-8

# # Predict titanic survival rate

# In[13]:


import pickle
import numpy as np
import streamlit as st
import pandas as pd


# In[14]:


# load the model from disk
loaded_model = pickle.load(open('Streamlit_Titanic.pkl', 'rb'))


# In[15]:


# Creating the Titles and Image

st.title("Predict passenger survival rate")
st.header("Predicting if a passenger would survive the titanic disaster based on a few attributes")


# In[16]:


# create the dropdown menus for input fields

DropDown1 = pd.DataFrame({'Sex': ['Male', 'Female']})
DropDown2 = pd.DataFrame({'Passenger class': ['1', '2', '3'],
                          'Port of Embark': ['Cherbourg','Queenstown','Southampton']})


# In[17]:


# take user inputs

Age = st.number_input('Age')
Fare = st.number_input('Fare')

Pclass = st.selectbox('Passanger Class',DropDown2['Passenger class'].unique())
temp_Sex = st.selectbox('Gender',DropDown1['Sex'].unique())
temp_Embarked = st.selectbox('Port of Embark', DropDown2['Port of Embark'].unique())

if temp_Sex == 'Male':
    Sex = 1
else:
    Sex = 0

if temp_Embarked == 'Cherbourg':
    Embarked = 1
else:
    if temp_Embarked == 'Queenstown':
        Embarked = 2
    else:
        Embarked = 3


# In[18]:


# store the inputs
features = [Age, Fare, Pclass, Sex, Embarked]

# convert user inputs into an array for the model
int_features = [int(x) for x in features]
final_features = [np.array(int_features)]

if st.button('Predict'):
    prediction = loaded_model.predict(final_features)
    st.balloons()
    if round(prediction[0],2) == 0:
        st.success('Unfortunately this passenger will not survive.')
    else:
        st.success('Luckily this passenger will survive.')


# In[ ]:


# save the notebook with python file extension on  
# C folder with name 'streamlitpredictsurvivalrate'

