import math
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier

train_data = np.genfromtxt('spambase_train.csv',delimiter=',')
[row,col] = np.shape(train_data)
x_train=train_data[:,0:col-2]
y_train=train_data[:,col-1]

test_data = np.genfromtxt('spambase_test.csv',delimiter=',')
[row1,col1] = np.shape(test_data)
x_test= test_data[:,0:col1-2]
y_test = test_data[:,col1-1]

results =[]
results1=[]
numTrees=[2,3,4,5,6,7,8,9,10,20,40,60]
cart = DecisionTreeClassifier()

for i in range (0,len(numTrees)):
    clf=BaggingClassifier(base_estimator=cart,n_estimators=numTrees[i],random_state=7).fit(x_train,y_train)
    scores = accuracy_score(y_test,clf.predict(x_test))
    results.append(100*(1-scores))
    clf1=AdaBoostClassifier(n_estimators=numTrees[i],random_state=7).fit(x_train,y_train)
    scores1 = accuracy_score(y_test,clf1.predict(x_test))
    results1.append(100*(1-scores1))

plt.plot(numTrees,results,color='blue',linestyle='--',label="Bagging")
plt.plot(numTrees,results1,color='red',label="AdaBoost")
plt.legend(["Bagging","Adaboost"])
plt.xlabel('No .of Classifiers')
plt.ylabel("Testing Error")
plt.show()
    
