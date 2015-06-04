from sklearn.linear_model import Perceptron
from sklearn import svm
import numpy as np
import pickle
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.cross_validation import KFold
import theanets

print( 'loading data')

with open('train_data.pkl' , 'rb') as fh:
    X = pickle.load(fh )

with open('train_labels.pkl' ,'rb') as fh:
    y = pickle.load(fh , encoding='bytes')

print( 'loaded')
print( ' +', X.shape[0], 'compounds')
print( ' +', y.shape[1], 'proteins')

protein = 2
print( ' + model of protein', (protein+1))
labels = y[:, protein].reshape(-1)
labels[labels == -1] = 0


for i in range(20):
    #X_train, X_test, labels_train, labels_test = train_test_split(X, labels, test_size=0.20, random_state=42)
    kf = KFold(len(labels), 5 , random_state = None)
    sumR = 0.0
    for train, test in kf:
        X_train, X_test, labels_train, labels_test = X[train], X[test], labels[train], labels[test]
        #clf = Perceptron(random_state=0, class_weight='auto', n_iter=50 + i*10)
        clf = KNN(n_neighbors= 1 + 2 *i)
        clf.fit(X_train, labels_train)

        labels_predict = clf.predict(X_test)
        
        
        sumR = sumR + sum( labels_predict == labels_test )/float(len(labels_test)) 
    print("KNN n_neighbors= =" ,  1+2*i , ':=' , sumR/5)

for i in range(40):
    #X_train, X_test, labels_train, labels_test = train_test_split(X, labels, test_size=0.20, random_state=42)
    kf = KFold(len(labels), 5 , random_state = None)
    sumR = 0.0
    for train, test in kf:
        X_train, X_test, labels_train, labels_test = X[train], X[test], labels[train], labels[test]
        #clf = Perceptron(random_state=0, class_weight='auto', n_iter=50 + i*10)
        clf = svm.SVC(max_iter = 50 + 20 *i)
        clf.fit(X_train, labels_train)

        labels_predict = clf.predict(X_test)
        
        
        sumR = sumR + sum( labels_predict == labels_test )/float(len(labels_test)) 
    print("SVC max_iter =" , 50 + 20 *i , ':=' , sumR/5)

for i in range(40):
    #X_train, X_test, labels_train, labels_test = train_test_split(X, labels, test_size=0.20, random_state=42)
    kf = KFold(len(labels), 5 , random_state = None)
    sumR = 0.0
    for train, test in kf:
        X_train, X_test, labels_train, labels_test = X[train], X[test], labels[train], labels[test]
        clf = Perceptron(random_state=0, class_weight='auto', n_iter=(1+i)*10)
        #clf = svm.SVC(max_iter = 50 + 20 *i)
        clf.fit(X_train, labels_train)

        labels_predict = clf.predict(X_test)
        
        
        sumR = sumR + sum( labels_predict == labels_test )/float(len(labels_test)) 
    print( "Perc m_iter =" , (1+i)*10 , ':=' , sumR/5)

kf = KFold(len(labels), 5 , random_state = None)
sumR = 0.0
for train, test in kf:
	X_train, X_test, labels_train, labels_test = X[train], X[test], labels[train], labels[test]
	
	clf = svm.SVC()
	clf.fit(X_train, labels_train)

	labels_predict = clf.predict(X_test)


	sumR = sumR + sum( labels_predict == labels_test )/float(len(labels_test)) 
print( "SVC max_iter = -1 :=" , sumR/5)