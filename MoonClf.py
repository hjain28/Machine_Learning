import numpy as np
import matplotlib.pyplot as plt

# import data 
from sklearn.datasets import make_moons
X, y = make_moons(n_samples=1000, noise=0.3, random_state=0)

# plot the original data
plt.figure()
plt.plot(X[np.where(y==1)[0], 0], X[np.where(y==1)[0], 1], 'o')
plt.plot(X[np.where(y==0)[0], 0], X[np.where(y==0)[0], 1], 's', c = 'r')
plt.show()

# build  a LR classifier
from sklearn.linear_model import LogisticRegression
clf = LogisticRegression()
clf.fit(X, y)

# generate some data to test the model; 
Xe, ye = make_moons(n_samples=1000, noise=0.3, random_state=0)
yhat_lr = clf.predict(Xe)

# clasify the data
plt.figure()
plt.plot(Xe[np.where(yhat_clf==1)[0], 0], Xe[np.where(yhat_clf==1)[0], 1], 'o')
plt.plot(Xe[np.where(yhat_clf==0)[0], 0], Xe[np.where(yhat_clf==0)[0], 1], 's', c = 'r')

