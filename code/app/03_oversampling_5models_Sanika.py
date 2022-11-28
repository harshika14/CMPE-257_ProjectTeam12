#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Importing the required libraries
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score
from imblearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer
from sklearn import metrics
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import pickle


# Here, I have read half_preprocessed file which is basically the preprocessed file which was saved and I have taken just the first half rows so it is easier to run it on the kernel.

# In[2]:


df =pd.read_csv("half_preprocessed.csv")
X = df['lemma']
y = df['review_stars']
df.head()


# In[3]:


X_try=X.tolist()
X_try=np.array(X_try)
X_try=X_try.reshape(-1,1)


# In[4]:


ros = RandomOverSampler(random_state=777)


# In[5]:


X_ROS, y_ROS = ros.fit_resample(X_try, y)


# In[6]:


X_train, x_test, Y_train, y_test = train_test_split(X_ROS,y_ROS,test_size=0.3,random_state=42)


# In[7]:


X_train= X_train.flatten()
x_test= x_test.flatten()


# ### Linear SVC

# In[8]:


svc_pipeline = newpipeline = Pipeline([('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()), 
    ('clf', LinearSVC())])
svc_pipeline.fit(X_train,Y_train)
print('Training set score: ' + str(svc_pipeline.score(X_train,Y_train)))
print('Test set score: ' + str(svc_pipeline.score(x_test,y_test)))
with open('svc_oversample.pickle', 'wb') as f:
        pickle.dump(svc_pipeline, f)
svc_yhat=svc_pipeline.predict(x_test)
svc_cm=confusion_matrix(y_test, svc_yhat)
svc_disp = ConfusionMatrixDisplay(confusion_matrix=svc_cm)
svc_disp.plot()
print(metrics.classification_report(y_test, svc_yhat))


# ### Random Forest 

# In[9]:


randomforest_pipeline = Pipeline([('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()), 
    ('clf', RandomForestClassifier())])
randomforest_pipeline.fit(X_train,Y_train)
print('Training set score: ' + str(randomforest_pipeline.score(X_train,Y_train)))
print('Test set score: ' + str(randomforest_pipeline.score(x_test,y_test)))
with open('svc_oversample.pickle', 'wb') as f:
    pickle.dump(randomforest_pipeline, f)
rf_yhat=randomforest_pipeline.predict(x_test)
rf_cm=confusion_matrix(y_test, rf_yhat)
rf_disp = ConfusionMatrixDisplay(confusion_matrix=rf_cm)
rf_disp.plot()
print(metrics.classification_report(y_test, rf_yhat))


# ### Adaboost

# In[10]:


adaboost_pipeline = Pipeline([('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()), 
    ('clf', AdaBoostClassifier())])
adaboost_pipeline.fit(X_train,Y_train)
print('Training set score: ' + str(adaboost_pipeline.score(X_train,Y_train)))
print('Test set score: ' + str(adaboost_pipeline.score(x_test,y_test)))
with open('adc_oversample.pickle', 'wb') as f:
    pickle.dump(adaboost_pipeline, f)
adc_yhat=adaboost_pipeline.predict(x_test)
adc_cm=confusion_matrix(y_test, adc_yhat)
adc_disp = ConfusionMatrixDisplay(confusion_matrix=adc_cm)
adc_disp.plot()
print(metrics.classification_report(y_test, adc_yhat))


# ### K-nearest Neighbours

# In[11]:


knn_pipeline = Pipeline([('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()), 
    ('clf', KNeighborsClassifier())])
knn_pipeline.fit(X_train,Y_train)
print('Training set score: ' + str(knn_pipeline.score(X_train,Y_train)))
print('Test set score: ' + str(knn_pipeline.score(x_test,y_test)))
with open('knn_oversample.pickle', 'wb') as f:
    pickle.dump(knn_pipeline, f)
knn_yhat=knn_pipeline.predict(x_test)
knn_cm=confusion_matrix(y_test, knn_yhat)
knn_disp = ConfusionMatrixDisplay(confusion_matrix=knn_cm)
knn_disp.plot()
print(metrics.classification_report(y_test, knn_yhat))


# ### XGBoost

# In[12]:


le = LabelEncoder()
y = le.fit_transform(y)
X_ROS, y_ROS = ros.fit_resample(X_try, y)
X_train, x_test, Y_train, y_test = train_test_split(X_ROS,y_ROS,test_size=0.3,random_state=42)
xgb_pipeline = Pipeline([('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()), 
    ('clf', xgb.XGBClassifier())])
X_train= X_train.flatten()
x_test= x_test.flatten()
xgb_pipeline.fit(X_train,Y_train)
print('Training set score: ' + str(xgb_pipeline.score(X_train,Y_train)))
print('Test set score: ' + str(xgb_pipeline.score(x_test,y_test)))
with open('xgboost_oversample.pickle', 'wb') as f:
    pickle.dump(xgb_pipeline, f)
xgb_yhat=xgb_pipeline.predict(x_test)
xgb_cm=confusion_matrix(y_test, xgb_yhat)
xgb_disp = ConfusionMatrixDisplay(confusion_matrix=xgb_cm)
xgb_disp.plot()
print(metrics.classification_report(y_test, xgb_yhat))


# ## Comparison between models

# In[13]:


f1_scores = [0.89, 0.93, 0.49, 0.66, 0.89]
models = ['SVC', 'Random Forest', 'Adaboost', 'KNN', 'XGBoost']
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.bar(models,f1_scores)
plt.show()


# ### Here, from the above graph Random forest gives good results with good F-1 score.
