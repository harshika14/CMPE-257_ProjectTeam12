import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline, FeatureUnion
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score,precision_score,recall_score,classification_report,confusion_matrix, ConfusionMatrixDisplay
import pickle
def get_metrics(pipe):
    test_data=pd.read_csv("datasets/testing_data_rating.csv")
    st.write('Accuracy of this model ' + str(pipe.score(test_data["text"].to_list(),test_data["rating"].to_list())))
    st.write("Classification Report : ")
    report=classification_report(test_data["rating"].to_list(),pipe.predict(test_data["text"].to_list()),output_dict=True)
    df = pd.DataFrame(report).transpose()
    st.write(df)
    st.write("Confusion Matrix : ")
    cm=confusion_matrix(test_data["rating"].to_list(),pipe.predict(test_data["text"].to_list()))
    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    ax.matshow(cm, cmap=plt.cm.Blues, alpha=0.3)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(x=j, y=i,s=cm[i, j], va='center', ha='center', size='large')
    plt.xlabel('Predictions', fontsize=15)
    plt.ylabel('Actuals', fontsize=15)
    plt.title('Confusion Matrix', fontsize=15)
    st.pyplot(fig)

def app():
    ### this would give you the title on the browser tab
    st.set_page_config(page_title="Rating Demo")
    option = st.selectbox("Please pick the model you want to test.",('Linear Support Vector','Linear Support Vector with Oversampling'))
    if option == 'Linear Support Vector':
        with open('models/svcpipe.pickle', 'rb') as f:
            pipe = pickle.load(f)
    elif option == 'Linear Support Vector with Oversampling':
        with open('models/svc_oversample.pickle', 'rb') as f:
            pipe = pickle.load(f)
    st.header("Rating Prediction using" + option)
    review=st.text_input("Enter your review for the prediction")
    st.write("Your review is:",review)
    
    if st.button("Predict"):
        pred=pipe.predict([review])
        st.write("Your review is rated:",pred[0])
    
    get_metrics(pipe)

        
if __name__=='__main__':
    app()
