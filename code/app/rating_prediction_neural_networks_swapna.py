#!/usr/bin/env python
# coding: utf-8

# In[23]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import tensorflow_hub as hub
import tensorflow as tf
import os
from sklearn.model_selection import train_test_split


# In[5]:


df=pd.read_csv('california_restaurants.csv')


# In[6]:


df.columns


# In[19]:


pdf=pd.read_csv('preprocessed.csv')


# In[20]:


pdf


# In[18]:


df


# In[8]:


df=df[['text','review_stars']]


# In[21]:


ldf=pdf[['lemma','review_stars']]


# In[12]:


from sklearn.utils import class_weight
class_weights = list(class_weight.compute_class_weight(class_weight='balanced',
                                classes=np.unique(df['review_stars']),
                                y =df['review_stars']))


# In[57]:


X_train, X_test = train_test_split(df, test_size=0.2, random_state=42)


# In[58]:


X_train


# In[59]:


X_train['review_stars']=X_train["review_stars"]-1


# In[60]:


X_test['review_stars']=X_test["review_stars"]-1


# In[61]:


class_weights.sort()


# In[62]:


df['review_stars'].value_counts()


# In[63]:


class_weights


# In[16]:


weights={}
for index,weight in enumerate(class_weights):
    weights[index]=weight


# In[17]:


weights


# In[64]:


dataset_train=tf.data.Dataset.from_tensor_slices((X_train["text"].values,X_train["review_stars"].values))
dataset_test=tf.data.Dataset.from_tensor_slices((X_test["text"].values,X_test["review_stars"].values))
print("Len of training data",len(dataset_train))
print("Len of testing data",len(dataset_test))


# In[65]:


for text,target in dataset_train.take(1):
    print('Review: {},\n Star: {}'.format(text,target))


# In[66]:


for text,target in dataset_test.take(1):
    print('Review: {},\n Star: {}'.format(text,target))


# In[67]:


def fetch(text, labels):
    return text,tf.one_hot(labels,5)


# In[68]:


train_data_one=dataset_train.map(fetch)
test_data_one=dataset_test.map(fetch)


# In[69]:


next(iter(train_data_one))


# In[70]:


next(iter(test_data_one))


# In[71]:


embedding="https://tfhub.dev/google/tf2-preview/nnlm-en-dim128/1"


# In[73]:


train_data,train_labels=next(iter(train_data_one.batch(5)))


# In[72]:


hub_layer=hub.KerasLayer(embedding,output_shape=[128],input_shape=[],
                        dtype=tf.string,trainable=True)


# In[74]:


hub_layer(train_data[:1])


# In[75]:


model=tf.keras.Sequential()
model.add(hub_layer)
for units in [128,128,64,32]:
    model.add(tf.keras.layers.Dense(units,activation='relu'))
    model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.Dense(5,activation='softmax'))


# In[76]:


model.summary()


# In[77]:


model.compile(optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy'])


# In[78]:


train_data_one=train_data_one.shuffle(50000).batch(512)
test_data_one=test_data_one.batch(512)


# In[79]:


history=model.fit(train_data_one,
                    epochs=10,
                        validation_data=test_data_one,
                        class_weight=weights)


# In[81]:


results=model.evaluate(dataset_test.map(fetch).batch(20000),verbose=2)


# In[82]:


test_data,test_labels=next(iter(dataset_test.map(fetch).batch(30000)))


# In[84]:


y_pred=model.predict(test_data)


# In[85]:


from sklearn.metrics import confusion_matrix,classification_report


# In[87]:


print(classification_report(test_labels.numpy().argmax(axis=1),y_pred.argmax(axis=1)))


# In[ ]:




