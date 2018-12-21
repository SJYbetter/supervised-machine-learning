import os
import numpy as np
import scipy.io
from sklearn import svm




def read(filename):
    result = scipy.io.loadmat(filename)
    return np.matrix(result['X_trn'], dtype=np.float), np.matrix(result['Y_trn'],dtype=np.int32), np.matrix(result['X_tst'],dtype=np.float), np.matrix(result['Y_tst'],dtype=np.int32)

def precess(y):
    for i in range(y.size):
        if y[i] == 0:
           y[i] = -1
    y = np.asmatrix(data=y, dtype=np.int64)
    return y

def ith_Error(x, y, alphas, b, i):
    m, n = np.shape(x)
    e = 0
    for k in range(m):
        e += alphas[k]*y[k,0]*np.dot(x[i,:], x[k,:].T)
    error = e + b - y[i,0]
    return error


def class_error(x,y,w,b):
    m,n = np.shape(x)
    count = 0
    for i in range(m):
        predict = np.dot(x[i,:],w.T)+b
        if predict * y[i] < 0:
            count += 1
    return count



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
            if (np.dot(y[i],error_i) < (-tol) and alphas[i] < C) or (np.dot(y[i], error_i) > tol and alphas[i] > 0):
                k = np.random.randint(m, size=2)
                if k[0] == i:
                    j = k[1]
                else:
                    j = k[0]
                error_j = ith_Error(x, y, alphas, b, j)
                alphas_old[i] = alphas[i]
                alphas_old[j] = alphas[j]

                if y[i, 0] != y[j, 0]:
                    bounds_l = max(0, alphas[j]-alphas[i])
                    bounds_h = min(C, C+alphas[j]-alphas[i])
                else:
                    bounds_l = max(0, alphas[j]+alphas[i]-C)
                    bounds_h = min(C, alphas[j]+alphas[i])

                if bounds_l == bounds_h:
                    continue

                u = 2 * np.dot(x[i, :], x[j, :].T) - np.dot(x[i, :], x[i, :].T) - np.dot(x[j, :], x[j, :].T)

                if u >= 0:
                    continue
                alphas[j] = alphas[j] - y[j] * (error_i - error_j)/u
                if alphas[j] > bounds_h:
                    alphas[j] = bounds_h
                elif alphas[j] < bounds_l:
                    alphas[j] = bounds_l
                else:
                    alphas[j] = alphas[j]

                if abs(alphas[j]-alphas_old[j]) < 0.00001:
                    continue

                alphas[i] += np.dot(np.dot(y[i,0], y[j,0]), (alphas_old[j]-alphas[j]))
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
    print(alphas)
    #print(b)
    return alphas, b



def sk_smo_error(x, y, c, tol, alpha, b):
    clf = svm.SVC(C = c, kernel='linear', tol = tol)
    clf.fit(x, y)
    sk_coef = clf.coef_
    sk_b = clf.intercept_
    w_optimation = np.zeros([1,x.shape[1]])

    for i in range(y.size):
        w_optimation += alpha[i]*y[i,0]*x[i,:]

    print(w_optimation, b)
    #x2 = np.zeros([x.shape[0],1])

    #for i in range(y.size):
        #x2[i,0] = -1*(w_optimation[0,0]+b)/w_optimation[0,1]

    error_sk = class_error(x,y,sk_coef,sk_b)
    error_smo = class_error(x,y,w_optimation,b)

    print("the sklearn_svm error is: %f " %(error_sk))
    print("the smo error is %f " %(error_smo))

    return sk_coef, sk_b, w_optimation


if __name__ == "__main__":
    data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'dataset'))

    X_trn, Label_trn, X_tst, Label_tst = read(os.path.join(data_dir, "data1.mat"))
    x_trn, label_trn, x_tst, label_tst = read(os.path.join(data_dir, "data2.mat"))

    Label_trn_new = precess(Label_trn)
    Label_tst_new = precess(Label_tst)
    label_trn_new = precess(label_trn)
    label_tst_new = precess(label_tst)

    a1, b1 = smo(X_trn, Label_trn_new, 100, 0.001, 50)
    #a2, b2 = smo(x_trn, label_trn_new, 100, 0.001, 50)
    #print('ssss', a1,b1)

    print("dataset1 trainning data")
    skc1, skb1, w_opt1 = sk_smo_error(X_trn, Label_trn_new, 1000, 0.01, a1, b1)

    #print("dataset1 testing data")
    #skc11, skb11, w_opt11 = sk_smo_error(X_tst, Label_tst_new, 1000, 0.01, a1, b1)

    #print("dataset2 trainning data")
    #skc2, skb2, w_opt2 = sk_smo_error(x_trn, label_trn_new, 100, 0.01, a2, b2)

    #print("dataset2 testing data")
    #skc22, skb22, w_opt22 = sk_smo_error(x_tst, label_tst_new, 100, 0.01, a2, b2)




