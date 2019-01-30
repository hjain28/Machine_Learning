import numpy as np
from sklearn import svm
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score

data =  np.genfromtxt('breast-cancer-wisc.csv',delimiter=',')
row,col = np.shape(data)
X = data[:,1:col-2];
Y = data[:, col-1];
clf1 = svm.SVC()
clf1.fit(X,Y)
scores1 = cross_val_score(clf1, X, Y, cv=5)
cverror1 = 1-scores1.mean()
print('5 fold cross validation error is %0.2f\n', cverror1)

pca = PCA(n_components=2)
Xnew = pca.fit_transform(X)
clf2 = svm.SVC()
clf2.fit(Xnew,Y)
scores2 = cross_val_score(clf1, Xnew, Y, cv=5)
cverror2 = 1-scores2.mean()
print('After PCA, 5 fold cross validation error is %0.2f\n', cverror2)


