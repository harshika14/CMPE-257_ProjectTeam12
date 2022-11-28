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
    test_data=pd.read_csv("datasets/test_data_tagging.csv")
    st.write('Accuracy of this model ' + str(pipe.score(test_data["text"].to_list(),test_data["categories"].to_list())))
    st.write("Classification Report : ")
    report=classification_report(test_data["categories"].to_list(),pipe.predict(test_data["text"].to_list()),output_dict=True)
    df = pd.DataFrame(report).transpose()
    st.write(df)
    st.write("Confusion Matrix : ")
    cm=confusion_matrix(test_data["categories"].to_list(),pipe.predict(test_data["text"].to_list()))
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
    label_dict={0:"Active Life", 1:"Automotive",2:"Beauty & Spas",3:"Restaurants",4:"Shopping"}
    st.set_page_config(page_title="Tagging Demo")
    option = st.selectbox("Please pick the model you want to test.",('Linear Support Vector','Naive Bayes','SVM with SGD'))
    if option == 'Linear Support Vector':
        with open('models/tagging_model.pickle', 'rb') as f:
            pipe = pickle.load(f)
    elif option == 'Naive Bayes':
        with open('models/tagging_model_nb.pickle', 'rb') as f:
            pipe = pickle.load(f)
    elif option == 'SVM with SGD':
        with open('models/tagging_model_sgd.pickle', 'rb') as f:
            pipe = pickle.load(f)
    st.header("Tagging Prediction using "+option)
    review=st.text_input("Enter your review for the prediction")
    st.write("Your review is:",review)
    
    if st.button("Predict"):
        pred=pipe.predict([review])
        st.write("Your review can be tagged as :",label_dict.get(pred[0]))
    
    get_metrics(pipe)
        
if __name__=='__main__':
    app()