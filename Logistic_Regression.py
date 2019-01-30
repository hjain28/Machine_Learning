import numpy as np
from math import exp

data = np.genfromtxt('spambase_train.csv', delimiter = ',')

X = data[:, 0:-1]
X0 = np.ones((len(X[:,1]),1))
X = np.hstack((X0, X))
y = data[:, -1]

lmb = 0.01

n = len(X[0])

def predict(row, w):
    return 1/(1 + exp((-1*((np.dot(w, row))))))

w_new = [0]*n

def LR(data, w_new):
    for m in range(0, 500): 
        for i in range(0, int(len(data)*1)):
            w_old = w_new
            y_p = predict(X[i], w_old)
            w_new = w_old + (lmb*(y[i] - y_p)*X[i])
            w = w_new - w_old
            if(np.sum(np.power(w, 2))<0.000001):
                return(w_new)
                
w_new = LR(data, w_new)
print(w_new)
