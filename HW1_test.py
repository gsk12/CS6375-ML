# Machine Learning Homework 1  
# Question 1

# a setup
import numpy as np
from matplotlib import pyplot as plt

# b Generate Synthetic Data 

n = 200
x = np.random.uniform(0.0, 6.0, n)

y = np.sin(x) + np.sin(2*x) + np.random.normal(0.0, 0.25, n)
# print(y)
# print(len(x))
# print(len(y))
x_true = np.arange(0.0, 6.0, 0.05) 
y_true = np.sin(x_true) + np.sin(2 * x_true)
#plt.plot(x_true , y_true , linestyle='-', linewidth=3.0, color='k')
#plt.plot(x, y, marker='o', markerfacecolor='r', markeredgecolor='k',linestyle='none')
#plt.show()

# c Create Training and test sets 

np.random.shuffle(x)
np.random.shuffle(y)

x_trn = x[:150]
y_trn = y[:150]
x_tst = x[150:]
y_tst = y[150:]

# print ('xtrn', x_trn)
# print ('ytrn', y_trn)
# print ('xtst', x_tst)
# print ('ytst', y_tst)
# print(len(x_trn))
# print(len(y_trn))
# print(len(x_tst))
# print(len(y_tst))

# another way to random sample
# from random import shuffle
# n = 10 #number of rows in your dataset
# indices = range(10)
# shuffle(indices)
# print "train indices:", indices[:6]
# print "validation indices:", indices[6:8]
# print "test indices:", indices[8:]

# d monomial basis functions

def transform_data(x, d):
    print ("Vandermonde matrix")
    print("x",x)
    print("d",d)
    phi = np.vander(x,d)
    print("phi",phi)
    return phi

# e training and testing

def train_model(phi, y):
    print ("training model")
    print("phi",phi)
    print("y",y)
    #ols = (np.linalg.inv(phi.transpose() @ phi)) * (phi.transpose() @ y)
    ols = np.linalg.pinv(phi) @ y
    print("ols",ols)
    return ols

def test_model(w, phi, y):
    print ("testing model")
    #mse = np.polyfit(phi, w)
    # print("w",w)
    # print("phi",phi)
    # print("y",y)
    print("wtran",(w.transpose() @ phi[0]) - y)
    mse = (((y) - (w.transpose() * phi)) ** 2).mean()
    return mse

for d in range(5, 16): 
    phi_trn = transform_data(x_trn, d) 
    print("phi_trn",phi_trn)
    w = train_model(phi_trn, y_trn)
    print("w",w)
    phi_tst = transform_data(x_tst, d)
    print("phi_tst",phi_tst)
    e_mse = test_model(w, phi_tst, y_tst)
    print("e_mse",e_mse)