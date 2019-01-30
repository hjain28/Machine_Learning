## import data from file
import numpy as np

data = np.genfromtxt('spambase_train.csv',delimiter=',')
features_train = data[:,0:-1]
label_train = data[:,-1]

## for classifiers
from sklearn.naive_bayes import GaussianNB

clf = GaussianNB()

## To measures accuracy of classifier 
from sklearn.model_selection import cross_val_score

scores = cross_val_score(clf, features_train, label_train, cv=5)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))


