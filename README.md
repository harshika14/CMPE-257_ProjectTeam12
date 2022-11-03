# CMPE-257_ProjectTeam12

Submitted by- Team 12

## Project title- “Business reviews and Data analysis using Machine learning on yelp.” <br>

### Team members- 

1.Sanika Vijaykumar Karwa  <br>
github username: sanika-karwa 
https://github.com/sanika-karwa

2.Harshika Shrivastava <br>
github username: harshika14
https://github.com/harshika14

3.Tirupati Venkata Sri Sai Rama Raju Penmatsa <br>
github username: rajuptvs
https://github.com/rajuptvs

4.Swapna Kotha <br>
github username: kothaswapna
https://github.com/kothaswapna

 
Note: In the repository, notebook containing preprocessing of data is attached. Also, a pdf file for better visualization of data with all the information is included in the repository. 
 
 
 
# About our Dataset: 
   For this project, we are using the Yelp dataset which can be accessed from here -
     Link to dataset:  1. https://www.kdnuggets.com/datasets/index.html
                                2. https://www.yelp.com/dataset
 We are using this Yelp dataset and performing the preprocessing on it to get clean data for our project.
 
 
# The motivation behind our Project:
Reviews are one of the most important assets for any business. They can be useful for any business for attracting new customers, increasing sales, and for understanding the scope of improvement for any business. From the customer's point of view, it helps customers to get a better user experience. We have decided to work on this project because utilize our machine learning models to help business owners as well as customers for a better experience.

For this project, we are using the yelp dataset. Yelp is an American company that provides an application and website where users can access the reviews given by other customers about any business. We are using an open dataset provided by Yelp.


# Problem Statements and our solution:
One of the problems that any user on Yelp faces is that there are reviews but these reviews are not filtered. They are not providing a clear and overall conclusion or any graphical representation of the information to the user. We will try to provide some representation to the user using our project.<br>
Another problem that we want to deal with in this project is - there can be fake reviews on yelp, we will try to identify a pattern in them and will try to detect fake reviews from our model. <br>
Thirdly, it is often common to get negative reviews as well as positive reviews. With our project implementation, we will try to draw conclusions from the reviews of the user and will try to analyze the sentiments of the user about any business using machine learning.
On top of that, we will try to provide the user with a visual representation (more like a graph or word cloud) of the most frequent keywords from the data about a particular business, so they do not have to go through all the reviews instead they can look at the keywords and get an idea about the reviews.


# Potential Methods:
For this project, we will be implementing supervised learning methods. We are trying to implement some functionality in our project like-
Classification of reviews using classification methods such as- TF-IDF VECTORIZER,Gradient Boosting classifier, Naive Bayes, Decision Tree, Neural Networks (probably a transformers if time permits)<br>
Reviews and the text data will be preprocessed further using the following techniques- NLTK library which includes tools for lemmitization, stemming and removing the unnecessary words <br>
Data visualization for business owners by comparing various other businesses in the same city. <br>
So, another column that has been observed is that there is a column for the open/close "status", probably which can be used to check if the customer ratings was the cause of the closure of that particular business.<br>

# PreProcessing Techniques:
In the Preprocessing, 
We have downloaded the datasets as a .json files and converted them to csv files respectively.<br>
Null values have been removed from the detected columns, the detected number of null values are fairly insignificant in comparision to the size of the dataset, so they have been removed. <br>
Using the sqldf library, we have gotten the top categories of business and their reviews have been merged.<br>
Using the Business Id as a unique identifier key, these csv's have been merged to combine the business dataset with the reviews associated to them. <br>
Data visualization have been plotted for checking the rate of reviews, no of reviews that are positive vs negative reviews- Interactive Plots using the Plotly library <br>
Feature Engineering to gain some extra info on the polarity of the text using the reviews<br>

# Some of the Challenges which we might face:

Dataset looks fairly clean, but the dataset has some special attributes, which can be further extracted, but currently due to massive size of the dataset, we are still observing the data further.<br>
Feature Engineering also would be my one of our challenge, as we would need a better understanding of the domain knowledge.<br>
Distribution of the rating's seems to indicate that majority of the ratings are positive or neutral at the most, this can make the model biased, this is something that needs to be further looked into while creating a train/test split to have an even split.<br>
Handling of the .ipynb notebooks would be difficult due to the vast size of the dataset -- Planning to split the .ipynb files for individual tasks <br>


# Some questions and conclusions 
With our project, we will try to answer some questions like-
What are the top reviews about this shop?
How many other related businesses are there in the same city?

What is the feedback for any business, are the reviews positive or negative?
