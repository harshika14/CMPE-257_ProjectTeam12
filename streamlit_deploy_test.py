import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.metrics import accuracy_score,precision_score,recall_score,classification_report
import pickle

def app():
    with open('svcpipe.pickle', 'rb') as f:
        pipe = pickle.load(f)
    st.header("Rating Prediction using Linear SVC")
    review=st.text_input("Enter your review for the prediction")
    st.write("Your review is:",review)
    
    if st.button("Predict"):
        pred=pipe.predict([review])
        st.write("Your review is rated:",pred[0])
        
if __name__=='__main__':
    app()
