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

# Creating the Bag of words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=1500)
X = cv.fit_transform(corpus).toarray()
y = df.iloc[:, -1].values

# Spliting the dataset
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Scores List
Acc_score = []
Pre_score = []
Rec_score = []
F1_score = []

# LOGISTIC REGRESSION
from sklearn.linear_model import LogisticRegression
classifier_lr = LogisticRegression(random_state=0)
classifier_lr.fit(X_train, y_train)
y_lr = classifier_lr.predict(X_test)
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
Acc_score.append(accuracy_score(y_test, y_lr))
Pre_score.append(precision_score(y_test, y_lr))
Rec_score.append(recall_score(y_test, y_lr))
F1_score.append(f1_score(y_test, y_lr))

# K-NEAREST NEIGHBORS
from sklearn.neighbors import KNeighborsClassifier
classifier_knn = KNeighborsClassifier(n_neighbors=7, metric='minkowski', p=2)
classifier_knn.fit(X_train, y_train)
y_knn = classifier_knn.predict(X_test)
Acc_score.append(accuracy_score(y_test, y_knn))
Pre_score.append(precision_score(y_test, y_knn))
Rec_score.append(recall_score(y_test, y_knn))
F1_score.append(f1_score(y_test, y_knn))

# KERNEL SUPPORT VECTOR MACHINE
from sklearn.svm import SVC
classifier_ksvm = SVC(kernel='rbf', random_state=0)
classifier_ksvm.fit(X_train, y_train)
y_ksvm = classifier_ksvm.predict(X_test)
Acc_score.append(accuracy_score(y_test, y_ksvm))
Pre_score.append(precision_score(y_test, y_ksvm))
Rec_score.append(recall_score(y_test, y_ksvm))
F1_score.append(f1_score(y_test, y_ksvm))

# NAIVE BAYES
from sklearn.naive_bayes import GaussianNB
classifier_nb = GaussianNB()
classifier_nb.fit(X_train, y_train)
y_nb = classifier_nb.predict(X_test)
Acc_score.append(accuracy_score(y_test, y_nb))
Pre_score.append(precision_score(y_test, y_nb))
Rec_score.append(recall_score(y_test, y_nb))
F1_score.append(f1_score(y_test, y_nb))

# DECISION TREE CLASSIFICATION
from sklearn.tree import DecisionTreeClassifier
classifier_dtc = DecisionTreeClassifier(criterion='entropy', random_state=0)
classifier_dtc.fit(X_train, y_train)
y_dtc = classifier_dtc.predict(X_test)
Acc_score.append(accuracy_score(y_test, y_dtc))
Pre_score.append(precision_score(y_test, y_dtc))
Rec_score.append(recall_score(y_test, y_dtc))
F1_score.append(f1_score(y_test, y_dtc))

# RANDOM FOREST CLASSIFICATION
from sklearn.ensemble import RandomForestClassifier
classifier_rfc = RandomForestClassifier(n_estimators=10, criterion='entropy', random_state=0)
classifier_rfc.fit(X_train, y_train)
y_rfc = classifier_rfc.predict(X_test)
Acc_score.append(accuracy_score(y_test, y_rfc))
Pre_score.append(precision_score(y_test, y_rfc))
Rec_score.append(recall_score(y_test, y_rfc))
F1_score.append(f1_score(y_test, y_rfc))

Models = ['Logistic Regression', 'K-NN', 'Kernel-SVM', 'Naive Bayes', 'Decision Tree Classification', 'Random Forest Classification']
lst = list(zip(Models, Acc_score, Pre_score, Rec_score, F1_score))

com_df = pd.DataFrame(lst, columns=['Models', 'Accuracy', 'Precision', 'Recall', 'F1 Score'])

print(com_df)

# False negatives are not as bad here for restaurant reviews as they would in, e.g., medical diagnosis. I guess the restaurant would be more interested in having as few false positives, as they would miss out on criticism and on how to improve. Best models with that in mind include RandomForest and SVM-rbf, with highest precision and F1 score.

# TP = # True Positives, TN = # True Negatives, FP = # False Positives, FN = # False Negatives

# Accuracy = (TP + TN) / (TP + TN + FP + FN)

# Precision = TP / (TP + FP)

# Recall = TP / (TP + FN)

# F1 Score = 2 * Precision * Recall / (Precision + Recall)