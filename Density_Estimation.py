import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors.kde import KernelDensity
from math import log


def gen_cb(N, a, alpha): 
    d = np.random.rand(N, 2).T
    d_transformed = np.array([d[0]*np.cos(alpha)-d[1]*np.sin(alpha),d[0]*np.sin(alpha)+d[1]*np.cos(alpha)]).T
    s = np.ceil(d_transformed[:,0]/a)+np.floor(d_transformed[:,1]/a)
    lab = 2 - (s%2)
    data = d.T
    return data, lab 

#generate training data
X1, y1 = gen_cb(250, .5, 0)
plt.plot(X1[np.where(y1==1)[0], 0], X1[np.where(y1==1)[0], 1], 'o')
plt.plot(X1[np.where(y1==2)[0], 0], X1[np.where(y1==2)[0], 1], 's', c = 'r')
plt.show()

# use gaussian kernel density estimation
blue_kde = KernelDensity(kernel='gaussian', bandwidth=0.2).fit(X1[np.where(y1==1)])
red_kde = KernelDensity(kernel='gaussian', bandwidth=0.2).fit(X1[np.where(y1==2)])
n_blue = len(X1[np.where(y1==1)])
n_red = len(X1[np.where(y1==2)])
blue_prior = log(n_blue)-log(250)
red_prior = log(n_red)-log(250)

#generate testing data, and implement calssifier
X2, y2 = gen_cb(5000, .5, 0)
blue_logdensity = blue_kde.score_samples(X2)
blue_w = blue_logdensity + blue_prior
red_logdensity = red_kde.score_samples(X2)
red_w = red_logdensity + red_prior

i = 0
while i < len(blue_w):
    if blue_w[i] > red_w[i]:
        y2[i] = 1
    else:
        y2[i] = 2
    i = i + 1

plt.figure()
plt.plot(X2[np.where(y2==1)[0], 0], X2[np.where(y2==1)[0], 1], 'o')
plt.plot(X2[np.where(y2==2)[0], 0], X2[np.where(y2==2)[0], 1], 's', c = 'r')
plt.show()

#generate training data
X3, y3 = gen_cb(5000, .25, 3.1415/4)
plt.figure()
plt.plot(X3[np.where(y3==1)[0], 0], X3[np.where(y3==1)[0], 1], 'o')
plt.plot(X3[np.where(y3==2)[0], 0], X3[np.where(y3==2)[0], 1], 's', c = 'r')
plt.show()

# use gaussian kernel density estimation
blue_kde2 = KernelDensity(kernel='gaussian', bandwidth=0.1).fit(X3[np.where(y3==1)])
red_kde2 = KernelDensity(kernel='gaussian', bandwidth=0.1).fit(X3[np.where(y3==2)])
n_blue2 = len(X3[np.where(y3==1)])
n_red2 = len(X3[np.where(y3==2)])
blue_prior2 = log(n_blue2)-log(5000)
red_prior2 = log(n_red2)-log(5000)

#generate testing data, and implement calssifier
X4, y4 = gen_cb(5000, .25, 3.1415/4)
blue_logdensity2 = blue_kde2.score_samples(X4)
blue_w2 = blue_logdensity2 + blue_prior2
red_logdensity2 = red_kde2.score_samples(X4)
red_w2 = red_logdensity2 + red_prior2

i = 0
while i < len(blue_w2):
    if blue_w2[i] > red_w2[i]:
        y4[i] = 1
    else:
        y4[i] = 2
    i = i + 1
    
plt.figure()
plt.plot(X4[np.where(y4==1)[0], 0], X4[np.where(y4==1)[0], 1], 'o')
plt.plot(X4[np.where(y4==2)[0], 0], X4[np.where(y4==2)[0], 1], 's', c = 'r')
plt.show()




