# -*- coding: utf-8 -*-
"""
Created on Mon Feb 27 05:20:58 2023

@author: barry
"""

import streamlit as st
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
#from sklearn.ensemble import Hist_Gradient_Boosting_Classifier

def main():
    st.title("Heart Health Predictor ")
    gif_url = "http://3.bp.blogspot.com/-nslbumdNSCM/TdDSGwrDOwI/AAAAAAAACcw/TbQGaL5JhnE/s1600/3d+gif+animation+free+blog....+beating+heart++doctor+cardiologist+heart+surgeon+.gif"
    st.image(gif_url, width=300, height=300) #use_column_width=True)
    
    # Upload CSV file
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    if uploaded_file is not None:
        # Convert CSV file to pandas dataframe
        df = pd.read_csv(uploaded_file)
        
        # Show the first 10 rows of the dataframe
        st.write("First 10 rows of the uploaded file:")
        st.write(df.head(10))
        
        # Split data into features and target
        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]
        
        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        # Create a logistic regression object and fit to training data
        logreg = LogisticRegression()
        logreg.fit(X_train, y_train)
        
        # Make predictions on testing data and calculate accuracy
        y_pred = logreg.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        # Show accuracy score
        st.write("Without Changes a Heart Attack is:", accuracy,"%" "Likely.")

if __name__ == "__main__":
    main()
