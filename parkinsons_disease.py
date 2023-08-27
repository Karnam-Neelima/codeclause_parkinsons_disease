import numpy as np 
import pandas as pd
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
      data_master = pd.read_csv("/kaggle/input/parkinsons-data-set/parkinsons.data")
data_master.head(5)
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
data_master.info()
for i in range(1,len(data_master.columns)-1):
    sns.boxplot(x='status',y=data_master.iloc[:,i],data=data_master,orient='v',ax=axes[i])
plt.tight_layout()
plt.show()
data_master.status.value_counts()
'''for dataset in data_master: 
    dataset['status'] = dataset['status'].astype(float) 

data_master['status'].value_counts()'''
X = data_master.drop(['status', 'name'], axis = 1)
y = data_master.status
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)

from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix

def clf_scores(clf, y_predicted):
    # Accuracy
    acc_train = clf.score(X_train, y_train)*100
    acc_test = clf.score(X_test, y_test)*100
    
    roc = roc_auc_score(y_test, y_predicted)*100 
    tn, fp, fn, tp = confusion_matrix(y_test, y_predicted).ravel()
    cm = confusion_matrix(y_test, y_predicted)
    correct = tp + tn
    incorrect = fp + fn
    
    return acc_train, acc_test, roc, correct, incorrect, cm
from sklearn.linear_model import LogisticRegression
clf_lr = LogisticRegression()
clf_lr.fit(X_train, y_train)

Y_pred_lr = clf_lr.predict(X_test)
print(clf_scores(clf_lr, Y_pred_lr))

from sklearn.neighbors import KNeighborsClassifier
clf_knn = KNeighborsClassifier(n_neighbors=3)
clf_knn.fit(X_train, y_train)

Y_pred_knn = clf_knn.predict(X_test)
print(clf_scores(clf_knn, Y_pred_knn))

from sklearn.naive_bayes import GaussianNB
clf_gnb = GaussianNB()
clf_gnb.fit(X_train, y_train)

Y_pred_gnb = clf_gnb.predict(X_test)
print(clf_scores(clf_gnb, Y_pred_gnb))

data_master.columns

from sklearn.preprocessing import StandardScaler

# copy of datasets
X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()

# numerical features
num_cols = ['MDVP:Fo(Hz)', 'MDVP:Fhi(Hz)', 'MDVP:Flo(Hz)', 'MDVP:Jitter(%)',
       'MDVP:Jitter(Abs)', 'MDVP:RAP', 'MDVP:PPQ', 'Jitter:DDP',
       'MDVP:Shimmer', 'MDVP:Shimmer(dB)', 'Shimmer:APQ3', 'Shimmer:APQ5',
       'MDVP:APQ', 'Shimmer:DDA', 'NHR', 'HNR', 'RPDE', 'DFA',
       'spread1', 'spread2', 'D2', 'PPE']

for i in num_cols:
    
    # fit on training data column
    
scale = StandardScaler().fit(X_train_scaled[[i]])
    
    # transform the training data column
    X_train_scaled[i] = scale.transform(X_train_scaled[[i]])
    
    # transform the testing data column
    X_test_scaled[i] = scale.transform(X_test_scaled[[i]])

X_train.describe()

from sklearn.linear_model import LogisticRegression
clf_lr = LogisticRegression()
clf_lr.fit(X_train_scaled, y_train)

Y_pred_lr = clf_lr.predict(X_test_scaled)
print(clf_scores(clf_lr, Y_pred_lr))

from sklearn.neighbors import KNeighborsClassifier
clf_knn = KNeighborsClassifier(n_neighbors=3)
clf_knn.fit(X_train_scaled, y_train)

Y_pred_knn = clf_knn.predict(X_test_scaled)
print(clf_scores(clf_knn, Y_pred_knn))

from sklearn.naive_bayes import GaussianNB
clf_gnb = GaussianNB()
clf_gnb.fit(X_train_scaled, y_train)

Y_pred_gnb = clf_gnb.predict(X_test_scaled)
print(clf_scores(clf_gnb, Y_pred_gnb))

from sklearn.svm import SVC

clf_svm = SVC()
clf_svm.fit(X_train_scaled, y_train)

Y_pred_svm = clf_svm.predict(X_test_scaled)
print(clf_scores(clf_svm, Y_pred_svm))

from mlxtend.classifier import StackingClassifier
from sklearn import model_selection

lr = LogisticRegression()
sclf = StackingClassifier(classifiers=[clf_knn, clf_svm, clf_gnb], 
                          meta_classifier=lr)
sclf.fit(X_train_scaled, y_train)
for clf, label in zip([clf_knn, clf_svm, clf_gnb, sclf], 
                      ['KNN', 
                       'SVM', 
                       'Naive Bayes',
                       'StackingClassifier']):

    Y_pred = clf.predict(X_test_scaled)
    scores = clf_scores(clf, Y_pred)
    
    print(scores, label)

X_train_scaled.describe()
