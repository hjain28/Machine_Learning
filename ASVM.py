import numpy as np
import matplotlib.pyplot as plt
import cvxopt as cp
from sklearn import svm
##
D = np.genfromtxt('source_train.csv', delimiter=',')
X = D[:,:-1]
y = D[:,-1]
D = np.genfromtxt('source_test.csv', delimiter=',')
X_test = D[:,:-1]
y_test = D[:,-1]
Ws = (svm.SVC(kernel='linear').fit(X,y)).coef_
Ws_mat=cp.matrix(Ws)
##
D = np.genfromtxt('target_train.csv', delimiter=',')
X = D[:,:-1]
y = D[:,-1]
B=1;
yT=y.transpose()
XT=cp.matrix(X.transpose())
WsT=cp.matrix(Ws.transpose())
X_mat=cp.matrix(X)
y_mat=cp.matrix(y)
train_len=len(y)
print(cp.matrix(np.reshape(y,(1,50))))
##
def compute_F(y_mat,X_mat,XT,WsT,train_len,i,B):
    Fi=[]
    la=-1*(1 - (B*y_mat[i,:]*X_mat[i,:]*WsT))
    Fi=np.append(Fi,la)
    return (Fi)
    
def compute_H(y_mat,X_mat,XT,WsT,train_len,i,B):
    Hi=[[]]
    for j in range(0,train_len):
        l=y_mat[i,:]*y_mat[j,:]*X_mat[i,:]*XT[:,j]
        Hi =np.append(Hi,l)
    return(Hi.flatten())
##
P=[]
q=[]
for k in range(0,train_len):
    m=compute_F(y_mat,X_mat,XT,WsT,train_len,k,B)
    l=compute_H(y_mat,X_mat,XT,WsT,train_len,k,B)
    q.append(m)
    P.append(l)
##   
q=np.array(q).tolist() 
P=np.array(P).tolist()
P=cp.matrix(P)
q=cp.matrix(np.matrix(q))
A=cp.matrix(np.reshape(y,(1,50)))
b=cp.matrix(0.67)
G0=np.identity(50)
G1=np.append(G0,G0,axis=0)
h1=np.full((50,1),0.0)
h2=np.full((50,1),1.0)
h1=np.append(h1,h2,axis=0)
##
G=cp.matrix(G1)
h=cp.matrix(h1)
sol=cp.solvers.qp(P,q,G,h,A,b)
alpha=sol['x']
##
s1=cp.matrix([[0],[0]])
for k in range(0,train_len):
    s1=s1 + A[k]*alpha[k]*X_mat[k,:]
Wt=Ws_mat*B + s1
##
D = np.genfromtxt('target_test.csv', delimiter=',')
X_test = D[:,:-1]
y_test = D[:,-1]
print(Wt[0],Wt[1])
##
accuracy=len(X_test)
for k in range(0,len(X_test)):
    predict=X_test[k,0]*Wt[0]+X_test[k,1]*Wt[1]
    if(predict*y_test[k]<0):
        accuracy=accuracy-1
print("accuracy = ",(accuracy/len(X_test))*100)
##
def predict(X_test,y_test,Wt):
    z=list()
    for k in range(0,len(X_test)):
        predict1=X_test[k,0]*Wt[0]+X_test[k,1]*Wt[1]
        if(predict1<0):
            z.append(0)
        else:
            z.append(1)
    Arr =  np.asarray(z)
    return Arr.flatten()
##
x = np.random.multivariate_normal([-1, -1], [[1, -.25], [-.25, 1]], 500).T
h = .02
x_min, x_max = x.min()+1 , x.max()+1
y_min, y_max = y.min() - 4, y.max() + 4
plt.title("Test Error graph with linear kernel")
'''plt.plot(X_test[np.where(y_test==1.0)[0], 0], X_test[np.where(y_test==1)[0], 1], 'o')
plt.plot(X_test[np.where(y_test==-1)[0], 0], X_test[np.where(y_test==-1)[0], 1], 's', c = 'r')'''
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Z = predict(np.c_[xx.ravel(), yy.ravel()],y_test,Wt)
print(Z)
Z = Z.reshape(xx.shape)
plt.pcolormesh(xx, yy, Z, cmap=plt.cm.Paired)

plt.show()

