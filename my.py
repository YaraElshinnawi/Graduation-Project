from concurrent.futures import thread
from datetime import date
from io import StringIO
import sys, os

from pandas._libs.json import dump
from sklearn.externals.funcsigs import signature
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, recall_score, f1_score, \
    precision_recall_curve, average_precision_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from mlxtend.plotting import plot_decision_regions
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.metrics import accuracy_score, precision_score
import string
from collections import Counter
import pandas as pd
import nltk
from sklearn.externals import joblib
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
from collections import Counter
import nltk
from matplotlib.pyplot import pie, axis, show
from pandas import Series,DataFrame
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from flask import Flask, render_template, flash, url_for
#from flask_mysqldb import mysql
import pymysql as MySQLdb

from flask import Flask
#
# app = Flask(__name__)
#
# @app.route('/', methods=['GET', 'POST'])
# def index():
#     return "Hello Nuclear Geeks"
#
# if __name__ == '__main__':
#     app.run()

from flask import Flask, render_template, request,session, url_for
from flask_mysqldb import MySQL
from werkzeug.utils import redirect

app = Flask(__name__,template_folder='templates')
app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = ''
app.config['MYSQL_DB'] = 'review'


    # def handle_sub_view(req):
    #     with app.test_request_context():
    #         from flask import request
    # if request.method == 'POST':
    #             result = request.form
    #             Review = result['Textt']
    #
    #             testReview['review']=str(Review)
    #             print(testReview['review'])
    #             Time = result['Time']
    #             Date = result['Date']
    #             UserID = result['UserID']
    #             HotelID = result['HotellID']
    #             cur = mysql.connection.cursor()
    #             cur.execute("insert into text (Text, Time, Date, UserID, HotelID) VALUES (%s,%s,%s,%s,%s)",(Review, Time, Date, UserID, HotelID))
    #             mysql.connection.commit()
    #             cur.close()
    #             return render_template("home.html", result=result)
mysql = MySQL(app)
testReview= pd.DataFrame()
count = lambda l1,l2: sum([1 for x in l1 if x in l2])
def punctCount(x):
    return count(x, set(string.punctuation))
def capsCount(x):
    sum = 0
    for char in x:
        sum+= char in "QWERTYUIOPASDFGHJKLZXCVBNM"
    return sum
# @app.route('/')
# def home():
#    return render_template('home.html')

@app.route('/')
def home():
 if not session.get('logged_in'):
  return render_template('login.html')
 else:

  return render_template('index.html', session_user_name =session['username'])
  #return index()



@app.route('/index', methods=['POST'])
def index():
    cur = mysql.connection.cursor()

    # cur = con.cursor()
    # cur.execute("SELECT * FROM hotel")
    # record = cur.fetchall()
    # for row in record:
    #     print("Id = ", row[0], )
    #     print("Name = ", row[1])
    #     image = row[2]

    cur.execute("SELECT * FROM hotel")

    rows = cur.fetchall();
    if request.method == 'POST':
        hotel1 = request.form['holiday']
        hotel2 = request.form['four']
        hotel3= request.form['hilton']
        if (hotel1)== 1:
            session['hotel']=1
            # return render_template('index.html',session_user_name =session['username'])
        elif (hotel2)==2:
            session['hotel'] = 2
            # return render_template('index.html', session_user_name=session['username'])
        elif (hotel3)==3:
            session['hotel'] = 3
            # return render_template('index.html', session_user_name=session['username'])


    # return render_template('index.html', rows=rows)
    return render_template('index.html',session_user_name =session['username'])




@app.route('/login', methods=['POST'])
def do_admin_login():

    error = None

    if request.method == 'POST':
        username_form  = request.form['username']
        password_form  = request.form['password']
        cur = mysql.connection.cursor()
        cur.execute("SELECT COUNT(1) FROM user WHERE fname = %s ;", [username_form]) # CHECKS IF USERNAME EXSIST
        if cur.fetchone()[0]:
            cur.execute("SELECT pass FROM user WHERE fname= %s AND pass = %s ;", ([username_form], [password_form])) # FETCH THE HASHED PASSWORD
            for row in cur.fetchall():
                if (password_form) == row[0]:
                    session['username'] = request.form['username']
                    session['logged_in'] = True

                    return home()

                else:
                    error = "Invalid Credential"
        else:
            error = "Invalid Credential"
            return render_template('login.html',error=error)



@app.route("/logout")
def logout():
  session['logged_in'] = False
  return home()
@app.route("/reviews")
def reviews():
    return render_template('home.html')

@app.route('/predict',methods = ['POST'])


def predict():

            data = pd.read_csv("Book1.csv")
            pd.set_option('display.expand_frame_repr', False)
            testvalue = pd.read_csv("Bb+.csv",delimiter = ';')


            df1 = data.groupby("Label").review
            data['TEXT_LENGTH'] = data['review'].apply(len)
#testReview['reviewlength']= testReview['review'].apply(len)
            c = data.groupby(["review"]).TEXT_LENGTH.agg(lambda x: sum(x) / len(x))
# #X = sum(data['TEXT_LENGTH'])/len(ll.max)
#
#
# avg = sum(len(word) for word in data.review)/len(data.review)
# print(data['TEXT_LENGTH'])



            data['caps_count'] = data['review'].apply(capsCount)
            caps = testReview.apply(capsCount)
            #
            # s = data.groupby('reviewerID').review.apply(list)
            # # data['s']= pd.DataFrame(s.loc[:'N','Y':].values)
            # # print(s)
            #
            # # s1 = data.groupby('reviewerID').emojis.apply(list)
            #
            # s1 = data.loc[[('reviewerID').review.apply(list)], ['emojis', 'punct_count', 'caps_count']]
            #
            # s2 = data.groupby('reviewerID').punct_count.apply(list)
            #
            # s3 = data.groupby('reviewerID').caps_count.apply(list)
            # # data['ss']= data.groupby(['s','s1','s2','s3','Label']).size()
            # pd.set_option('max_colwidth', 80000)
            # print(s1)


# data['tokenized_sents'] = data.apply(lambda row: nltk.word_tokenize(row['review']), axis=1)
# testReview['tokinize']=testReview.appl.apply(lambda row: nltk.word_tokenize(row['review']), axis=1)

# def tok(x):
#     sum = 0
#     for char in x:
#         sum+= len(str(data['tokenized_sents']))
#     return sum
# tokens = ll.apply(tok)
# print(tokens)


# print (Counter(data.review))

# sno = nltk.stem.SnowballStemmer('english')
# #s = "1   Let's try to be Good. 2   Being good doesn't make sense. 3   Good is always good."
# #s1 = str(data['tokenized_sents'])
# #d = pd.DataFrame()
# s2 = ll[0].apply(lambda x: sno.stem(x))
# counts =  Counter(s2)
# print(counts)



            data['punct_count'] = data['review'].apply(punctCount)

#testvalue['punct_count']=testvalue['review'].apply(punctCount)

            data["emojis"] = data["review"].apply(lambda x: 1 if ";)" in x.split() or ":)" in x.split() or ":-)" in x.split() else 0)

#testvalue['emojis']= testvalue['review'].apply(lambda x: 1 if ";)" in x.split() or ":)" in x.split() or ":-)" in x.split() else 0)


            stop = nltk.corpus.stopwords.words('english')
            user = data.groupby(['Date','reviewID','reviewerID','productID','Rating_1','Rating_2','Rating_3','Rating_4','emojis','punct_count','caps_count','Label']).size().unstack(fill_value=0)
            data['results'] = pd.DataFrame(user.loc[:'N','Y':].values)
            Labeled =data['Label'].replace(to_replace=['Y', 'N'], value=[1, 0])
            data['Labeled'] = Labeled

            # print(testReview)

# import re
# count = len(re.findall(r'\w+',str(data['tokenized_sents'].append(ll))))
# print (count)

#
# print ("The number of words in string are : " + str(data['tokenized_sents'].append(data['tokenized_sents'])))



#data.to_csv('Book1.csv')
            vec = CountVectorizer(ngram_range=(2,2),lowercase=False,stop_words=stop,max_features=4000 )
            X = vec.fit_transform(data.review)
            # Y = vectorizer.fit_transform(testReview)


            Train_X, Test_X,Train_Y, Test_Y = model_selection.train_test_split(X,data['results'],test_size=0.3,random_state=1)

# classifier = LogisticRegression()
# classifier.fit(Train_X, Train_Y)
# score = classifier.score(Test_X,Test_Y)*1000
# predictions_Log = classifier.predict(Y)
# print("Logistic Regression  Recall : ",recall_score(Test_Y,predictions_Log, average='weighted')*100 )
# print("F-score : ",f1_score(Test_Y, predictions_Log, average='weighted')*100)
# print("Logistic Regression  precision : ",precision_score(Test_Y, predictions_Log, average='weighted')*100)
# print("Logistic Regression Accuracy Score : ", score)
# print("Logistic Regression prediction : ", predictions_Log)
# print(confusion_matrix(Test_Y, predictions_Log))

            # Naive = naive_bayes.BernoulliNB()
            # Naive.fit(Train_X, Train_Y)
            SVM = svm.SVC(kernel='linear' ,C = 1 )
            SVM.fit(Train_X,Train_Y)
            # Naive.score(Test_X,Test_Y)
            # Alternative Usage of Saved Model
            # joblib.dump(SVM, 'NB_spam_model.pkl')
            # NB_spam_model = open('NB_spam_model.pkl','rb')
            # SVM = joblib.load(NB_spam_model)
            if request.method == 'POST':
                    message = request.form['message']
                    date =request.form['Date']
                    # reviewID=request.form['reviewID']
                    productID=1
                    reviewerID=session['username']
                    # label=request.form['label']
                    # rate1=request.form['r1ID']
                    # rate2 = request.form['r2ID']
                    # rate3 = request.form['r3ID']
                    # rate4 = request.form['r4ID']
                    testReview['Date']=[date]
                    # testReview['reviewID'] = [reviewID]
                    testReview['reviewerID'] = [reviewerID]
                    testReview['productID']=[productID]
                    # testReview['label']=[label]
                    # testReview['rating1'] = [rate1]
                    # testReview['rating2'] = [rate2]
                    # testReview['rating3'] = [rate3]
                    # testReview['rating4'] = [rate4]


                    testReview['review'] = [message]
                    cur = mysql.connection.cursor()
                    cur.execute("insert into text (Text, Date, UserID, HotelID) VALUES (%s,%s,%s,%s)", (message, date, reviewerID, productID))
                    mysql.connection.commit()
                    cur.close()
                    # testReview['reviewlength']= data.apply(len)
                    # testReview['reviewlength']= testReview['review'].apply(len)
                    # testReview['capscount'] = testReview['review'].apply(capsCount)
                    # testReview['punctcount'] = testReview['review'].apply(punctCount)
                    # testReview['emoji'] = testReview['review'].apply(
                    #     lambda x: 1 if ";)" in x.split() or ":)" in x.split() or ":-)" in x.split() else 0)
                    testReview.to_csv('Boook.csv')
                    # print(testReview)
                    # vec =  CountVectorizer(ngram_range=(3, 3), lowercase=False, stop_words=stop, max_features=40)
                    y = vec.transform(testReview.review).toarray()
                    # z = list(y.toarray())
                    # print(z.shape)
                    #y= vectorizer.fit_transform(testReview.review)
                    # z=list(y.toarray())
                    # print(z)
                    # vect = cv.transform(data).toarray()
                    # predictions_NB = Naive.predict(z)
                    predictions_SVM = SVM.predict(y)
                    print(predictions_SVM)
            return render_template('res.html', prediction=predictions_SVM)

if __name__ == '__main__':
    app.secret_key = os.urandom(12)
    app.run(debug=True)