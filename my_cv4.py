from datetime import date
from io import StringIO
import sys, os
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from mlxtend.plotting import plot_decision_regions
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.metrics import accuracy_score, precision_score
from sklearn.model_selection import train_test_split, GridSearchCV
import pandas
import self as self
from gensim.corpora import Dictionary
from nltk import ngrams, re, word_tokenize
import numpy as np
from numpy import array
import pandas as pd
from collections import defaultdict
from time import asctime
from nltk.corpus import stopwords
from scipy.sparse import csr_matrix
from sklearn import svm, model_selection, naive_bayes
from sklearn.feature_extraction.text import CountVectorizer
import csv
import nltk
from matplotlib.pyplot import pie, axis, show
from pandas import Series,DataFrame
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder

data = pd.read_csv("Book1.csv")
pd.set_option('display.expand_frame_repr', False)
testvalue = pd.read_csv("Bb+.csv",delimiter = ';')
#print(data.head(5))
ll = pd.DataFrame(data, columns = ['review'])
label = pd.DataFrame(data, columns = ['Label'])

df1 = data.groupby("Label").review
data['TEXT_LENGTH'] = data['review'].apply(len)
testvalue['TEXT_LENGTH']= testvalue['review'].apply(len)
#print(testvalue['TEXT_LENGTH'])
c = data.groupby(["Label"]).TEXT_LENGTH.agg(lambda x: sum(x) / len(x))

def capsCount(x):
    sum = 0
    for char in x:
        sum+= char in "QWERTYUIOPASDFGHJKLZXCVBNM"
    return sum
data['caps_count'] = data['review'].apply(capsCount)
testvalue['caps_count'] = testvalue['review'].apply(capsCount)

import string
count = lambda l1,l2: sum([1 for x in l1 if x in l2])
def punctCount(x):
    return count(x, set(string.punctuation))
data['punct_count'] = data['review'].apply(punctCount)
testvalue['punct_count']=testvalue['review'].apply(punctCount)

data["emojis"] = data["review"].apply(lambda x: 1 if ";)" in x.split() or ":)" in x.split() or ":-)" in x.split() else 0)
testvalue['emojis']= testvalue['review'].apply(lambda x: 1 if ";)" in x.split() or ":)" in x.split() or ":-)" in x.split() else 0)


stop = nltk.corpus.stopwords.words('english')
#data['stopwords'] = data['review'].apply(lambda x: [word for word in x.split() if word not in (stop)])
#print(data['stopwords'])
#data['tokenized_sents'] = data.apply(lambda row: nltk.word_tokenize(row['stopwords']), axis=1)
#data['trigram'] = data['stopwords'].apply(lambda x : list(ngrams(x, 3)))
#word_Final = data['trigram']
user = data.groupby(['Date','reviewID','reviewerID','productID','Rating_1','Rating_2','Rating_3','Rating_4','emojis','punct_count','caps_count','Label']).size().unstack(fill_value=0)
data['FakeReviewsData'] = pd.DataFrame(user.loc[:,'Y'].values)
data['RealReviewsData'] = pd.DataFrame(user.loc[:,'N'].values)
Labeled =data['Label'].replace(to_replace=['Y', 'N'], value=[1, 0])
Labls =testvalue['Label'].replace(to_replace=['Y', 'N'], value=[1, 0])

data['Labeled'] = Labeled
testvalue['labls'] = Labls
print(user)
#testvalue.to_csv('Bb.csv')

vectorizer = TfidfVectorizer(ngram_range=(2,2),lowercase=False,stop_words=stop,max_features=278)
#for index,row in data.iterrows():
# data['tri1']=[[data['trigram']]]
# print(data['trigram'].shape)
# print(data['tri1'].shape)

X = vectorizer.fit_transform(data.review)
Y = vectorizer.fit_transform(testvalue.review)
#print(vectorizer.get_feature_names())
# print(X.shape)
# # print(data['Labeled'].shape)
# print(Y.shape)

Train_X, Test_X,Train_Y, Test_Y = model_selection.train_test_split(X,data['Labeled'],test_size=0.3,random_state=1)

classifier = LogisticRegression()
classifier.fit(Train_X, Train_Y)
score = classifier.score(Test_X,Test_Y)*100
predictions_Log = classifier.predict(Y)
print("Logistic Regression prediction", predictions_Log)
print("Logistic Regression Accuracy Score -> ", score)


Naive = naive_bayes.BernoulliNB()
Naive.fit(Train_X, Train_Y)
predictions_NB =Naive.predict(Y)
print("Naive predection",predictions_NB)
print("Naive Bayes Accuracy Score -> ",Naive.score(Test_X, Test_Y)*100)


SVM = svm.SVC(kernel='linear' ,C = 1 )
SVM.fit(Train_X,Train_Y)
predictions_SVM = SVM.predict(Y)
print("SVM predection",predictions_SVM)
print("SVM Accuracy Score -> ",(SVM.score(Test_X, Test_Y)*100))


model = KNeighborsClassifier(n_neighbors=5)
model.fit(Train_X,Train_Y)
predictions_KNN = model.predict(Y)
print("KNN prediction",predictions_KNN)
print("KNN Accuracy score->",model.score(Test_X, Test_Y)*100)