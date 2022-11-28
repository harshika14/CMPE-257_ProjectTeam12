#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt # plotting
import numpy as np # linear algebra
import os # accessing directory structure
import pandas as pd 
import plotly.express as px
from pandasql import sqldf


# In[2]:


yelp_academic_dataset_business_json_path = '/home/016037047/yelp_data/yelp_academic_dataset_business.json'
yelp_business_dataset_json = pd.read_json(yelp_academic_dataset_business_json_path, lines=True)
#printing the overview of the business dataset
print(yelp_business_dataset_json.shape)
print('No of records in business dataset',yelp_business_dataset_json.shape[0])
print('No of features in business dataset',yelp_business_dataset_json.shape[1])
yelp_business_dataset_json.head()


# In[3]:


yelp_business_dataset_json.categories.value_counts()[:10]


# In[4]:


df_categories = yelp_business_dataset_json.assign(categories = yelp_business_dataset_json.categories.str.split(', ')).explode('categories')
df_categories.sample(3)


# In[5]:


df_categories.categories.value_counts()[:20]


# In[6]:


df_categories=df_categories.dropna()
df_categories.isna().sum()


# In[7]:


df_restaurant=df_categories[df_categories.categories == 'Restaurants'].head(5000)
df_shopping=df_categories[df_categories.categories == 'Shopping'].head(5000)
df_activelife=df_categories[df_categories.categories == 'Active Life'].head(5000)
df_beauty=df_categories[df_categories.categories == 'Beauty & Spas'].head(5000)
df_auto=df_categories[df_categories.categories == 'Automotive'].head(5000)


# In[8]:


multi_frame=[df_restaurant,df_shopping,df_activelife,df_beauty,df_auto]


# In[9]:


df_multicat=pd.concat(multi_frame)


# In[10]:


df_multicat.shape


# In[11]:


yelp_academic_review_json_path = '/home/016037047/yelp_data/yelp_academic_dataset_review.json'


# In[12]:


size = 600000
review = pd.read_json(yelp_academic_review_json_path, lines=True,
                      dtype={'review_id':str,'user_id':str,
                             'business_id':str,'stars':int,
                             'date':str,'text':str,'useful':int,
                             'funny':int,'cool':int},
                      chunksize=size)
chunk_list = []

for chunk_review in review:
    chunk_review = chunk_review.drop(['review_id','useful','funny','cool'], axis=1)
    chunk_review = chunk_review.rename(columns={'stars': 'review_stars'})
    chunk_merged = pd.merge(df_multicat, chunk_review, on='business_id', how='inner')
    print(f"{chunk_merged.shape[0]} out of {size:,} related reviews")
    chunk_list.append(chunk_merged)
multicat_review = pd.concat(chunk_list, ignore_index=True, join='outer', axis=0)


# In[15]:


df=multicat_review[["text","date","categories","review_stars"]]


# In[16]:


df


# In[18]:


df


# In[45]:


df_sample=df.sample(frac=1).reset_index(drop=True)
df_sample.shape
df_sample.categories.value_counts()


# In[68]:


df_test=df_sample[0:564718]
df_test.categories.value_counts()


# ### Had to use some other conditions to balance the dataset, as  dataframe operations were taking a lot of time 

# In[110]:


to_remove = np.random.choice(df_test[df_test['categories']=="Restaurants"].index,size=198000,replace=False)
df_testv1=df_test.drop(to_remove)


# In[111]:


df_testv1.categories.value_counts()


# In[112]:


to_remove = np.random.choice(df_testv1[df_testv1['categories']=="Restaurants"].index,size=8000,replace=False)
df_testv2=df_testv1.drop(to_remove)


# In[108]:


df_testv1.shape


# ### Basically ran the below code for almost all the values to bring them close to each other

# In[101]:


categories=df_testv2.categories.unique()


# In[102]:


categories


# In[118]:


to_remove = np.random.choice(df_testv2[df_testv2['categories']=="Active Life"].index,size=500,replace=False)
df_testv2=df_testv2.drop(to_remove)


# In[119]:


df_testv2.categories.value_counts()


# In[120]:


categories=['Beauty & Spas', 'Automotive', 'Active Life','Restaurants']


# In[121]:


# creating a for loop to remove the data as all of the categories listed have almost similar values
for i in categories:
    to_remove = np.random.choice(df_testv2[df_testv2['categories']==i].index,size=6000,replace=False)
    df_testv2=df_testv2.drop(to_remove)


# In[122]:


df_testv2.categories.value_counts()


# In[123]:


df_testv2.to_csv("tagging_dataset.csv")


# In[124]:


df_testv2.shape

