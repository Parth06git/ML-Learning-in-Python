# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
df = pd.read_csv('Restaurant_Reviews.tsv', delimiter='\t', quoting=3) # quoting = 3 is for ignoring all quote

# Cleaning the texts
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus = []
for i in range(1000):
    review = re.sub('[^a-zA-Z]', ' ', df['Review'][i])
    review = review.lower().split()
    ps = PorterStemmer()
    all_stopwords = stopwords.words('english')
    all_stopwords.remove('not')
    review = [ ps.stem(word) for word in review if not word in set(all_stopwords) ]
    review = ' '.join(review)
    corpus.append(review)

# print(corpus)

# Creating the Bag of words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=1500)
X = cv.fit_transform(corpus).toarray()
y = df.iloc[:, -1].values

# print(len(X[0])) # we are getting 1566 columns means there are 1566 words from all reviews. Not all words are needed so we can limit our columns to 1500 in defining the cv.

# Spliting the dataset
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Training the naive bayes model
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# Predicting the text result
y_pred = classifier.predict(X_test)

# Making the Confusion matrix
from sklearn.metrics import confusion_matrix, accuracy_score
print(confusion_matrix(y_test, y_pred))
print(accuracy_score(y_test, y_pred))

# Predicting if a single review is positive or negative
new_review = 'I hate this restaurant soo much'
new_review = re.sub('[^a-zA-Z]', ' ', new_review)
new_review = new_review.lower().split()
new_review = [ps.stem(word) for word in new_review if not word in set(all_stopwords)]
new_review = ' '.join(new_review)
new_y_pred = classifier.predict(cv.transform([new_review]).toarray())
print(new_y_pred)
