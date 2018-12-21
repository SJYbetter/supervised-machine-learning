import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.datasets import make_blobs



def read(filename):
    result = scipy.io.loadmat(filename)
    return result['X_trn'], result['Y_trn'], result['X_tst'], result['Y_tst']

def ith_Error(x, y, alphas, b, i):
    m,n = np.shape(x)
    F=0
    for k in range(m):
        f = np.dot(np.dot(np.dot(alphas[k],y[k]), x[k,:]), x[i,:].T)
        F += f
    return F + b - y[i]


def smo(x, y, C, tol, maxpass):
    m, n = np.shape(x)
    passes = 0
    b = 0.0
    alphas = np.zeros([m,1])
    alphas_old = np.zeros([m,1])

    while passes < maxpass:
        num_changed_alphas = 0
        for i in range(m):
            error_i = ith_Error(x, y, alphas, b, i)
            if (y[i]*error_i < (-tol) and alphas[i] < C) or (y[i]*error_i > tol and alphas[i] > 0):
                k = np.random.randint(m, size=2)
                if k[0] == i:
                    j = k[1]
                else:
                    j = k[0]
                error_j = ith_Error(x, y, alphas, b, j)
                alphas_old[i] = alphas[i]
                alphas_old[j] = alphas[j]

                if y[i] != y[j]:
                    bounds_l = max(0, alphas[j]-alphas[i])
                    bounds_h = min(C, C+alphas[j]-alphas[i])
                else:
                    bounds_l = max(0, alphas[j]+alphas[i]-C)
                    bounds_h = min(C, alphas[j]+alphas[i])

                if bounds_l == bounds_h:
                    continue
                u = 2 * np.dot(x[i,:],x[j,:].T) - np.dot(x[i,:],x[i,:].T) - np.dot(x[j,:],x[j,:].T)
                print('u', u)
                if u >= 0:
                    alphas[j] -= y[j]*(error_i - error_j)/u
                    if alphas[j] > bounds_h:
                        alphas[j] = bounds_h
                    elif alphas[j] < bounds_l:
                        alphas[j] = bounds_l
                    else:
                        alphas[j] = alphas[j]

                if abs(alphas[j]-alphas_old[j]) < 0.00001:
                    continue
                
                alphas[i] += np.dot(np.dot(y[i], y[j]), (alphas_old[j]-alphas[j]))
                b1 = b - error_i -y[i]*np.dot((alphas[i]-alphas_old[i]),np.dot(x[i,:],x[i,:].T)) - y[j]*np.dot((alphas[j]-alphas_old[j]),np.dot(x[i,:],x[j,:].T))
                b2 = b - error_j -y[i]*np.dot((alphas[i]-alphas_old[i]),np.dot(x[i,:],x[j,:].T)) - y[j]*np.dot((alphas[j]-alphas_old[j]),np.dot(x[j,:],x[j,:].T))
                if alphas[i] < C and alphas[i] > 0:
                    b = b1
                elif alphas[j] < C and alphas[i] > 0:
                    b = b2
                else:
                    b = 0.5*(b1+b2)
                num_changed_alphas += 1
        if num_changed_alphas == 0:
            passes += 1
        else:
            passes = 0
    #print(alphas)
    print(b)
    return alphas, b

if __name__ == "__main__":
    X_trn, Label_trn, X_tst, Label_tst = read("dataset/data1.mat")
    a,b = smo(X_trn, Label_trn, 1000000, 0.01, 10)
    print(a)

    x_trn, label_trn, x_tst, label_tst = read("dataset/data2.mat")
    c,d = smo(x_trn, label_trn, 1000, 0.01, 10)

