# Natural-language-processing-for-email-classification-using-the-Naive-Bayes-classifier-
Written in Python. Modules used are: pandas numpy matplotlib.pyplot seaborn CountVectorizer sklearn.naive_bayes_MultinomialNB sklearn.model_selection _train_test_split sklearn.metrics_classification_report, confusion_matrix
# Naive Bayes : Intuition
# Naive Bayes is a classification technique based on Bayes' Theorem
# Let's assume that you are data scientist working on major bank in NYC and you want to classify a new
# client as eligible to retire or not.
# Customer features are his/her age and salary.

# Importing our libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix


# Import Dataset
spam_df = pd.read_csv('emails.csv')
# printing the dataframe information
print('spam_df.head:\n', spam_df.head(10),'\n')
print('spam_df.tail:\n', spam_df.tail(10), '\n')
print('Describing the dataset:\n',spam_df.describe(),'\n')
print('Dataset information:\n', spam_df.info(),'\n')

# Visualizing the dataset
ham = spam_df[spam_df['spam']==0] # grouping the dataset into a ham group
print('Ham:\n', ham,'\n')

spam = spam_df[spam_df['spam']==1] # grouping the dataset into a spam group
print('Spam:\n', spam)

print('Spam percentage:\n', (len(spam)/len(spam_df))*100,'%')
print('Ham percentage:\n', (len(ham)/len(spam_df))*100,'%')

# Using the Countplot on seaborn to visualize the dataset
sns.countplot(spam_df['spam'],label = 'Count spam vs Ham')
plt.show()

# Using the count vectorizer to convert the text into numbers
# I am using the simple example below to illustrate a simple countvectorizer algorithm.
# sample = ['My name is Obierefu Finbarr Chukwuka', 'Obierefu is my surname','Finbarr is my first name',
#           'Chukwuka is my middle name','Your can also call me Chuka']
# sample_vectorizer = CountVectorizer()
# vectorized_sample = sample_vectorizer.fit_transform(sample)
# print(vectorized_sample.toarray())
# print (sample_vectorizer.get_feature_names())

# APPLYING COUNTVECTORIZER TO OUR SPAM/HAM DATASET
vectorizer = CountVectorizer()
spamham_countvectorizer = vectorizer.fit_transform(spam_df['text'])
print(vectorizer.get_feature_names())
print(spamham_countvectorizer.toarray())

# The shape of the spam_ham vectorizer
print(spamham_countvectorizer.shape)

# Training the Model
label = spam_df['spam'].values
print(label)

NB_classifier = MultinomialNB()
NB_classifier.fit(spamham_countvectorizer,label)


# Testing the Naive bayes classifer with a random sample
test_sample = ['Free Money!!!', 'Hi Finbarr, let me know if you need more information concerning the model']
test_sample_countvectorizer = vectorizer.transform(test_sample)
test_predict = NB_classifier.predict(test_sample_countvectorizer)
print(test_predict)

test_sample_1 = ['Hi Chisom, I will be at the airport in 3 hours, please don\'t hesitate to come pick me',
                 'Free sex!!!']
test_sample_countvectorizer = vectorizer.transform(test_sample_1)
test_predict = NB_classifier.predict(test_sample_countvectorizer)
print(test_predict)

# Divide the data into training and testing prior to training
X = spamham_countvectorizer
y = label

print (X.shape)
print(y.shape)

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)
NB_classifier = MultinomialNB()
NB_classifier.fit(X_train,y_train)

# Evaluating the model
# For the training sample( Not a good method to evaluate the model since we evaluate the model
# using the testing sample)
y_predict_train = NB_classifier.predict(X_train)
print(y_predict_train)

cm = confusion_matrix(y_train,y_predict_train)
sns.heatmap(cm, annot= True)
plt.show()

# For the testing sample
y_predict_test = NB_classifier.predict(X_test)
print(y_predict_test)

cm = confusion_matrix(y_test,y_predict_test)
sns.heatmap(cm,annot=True)
plt.show()

print('\n Classification report: \n', classification_report(y_test,y_predict_test))
