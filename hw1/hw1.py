import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv
import matplotlib.patches as mpatches
from prettytable import PrettyTable

x_inp = np.random.uniform(0.0, 6.0, 200) # training input



def hypothesis(input):
    value = []
    n = 1
    for i in input:
        value.append(np.sin(i) + np.sin(2 * i) + np.random.normal(0.0, 0.25))
        n=+1
    return value

y_out = hypothesis(x_inp)  # training output

x_true = np.arange(0.0, 6.0, 0.05)
y_true = np.sin(x_true) + np.sin(2*x_true)


plt.plot(x_true, y_true, linestyle='-', linewidth=3.0, color='k')

for i in range(200):
    plt.plot(x_inp[i], y_out[i], marker='o', markerfacecolor='r', markeredgecolor='k', linestyle='none')
plt.xlabel('x')
plt.ylabel('y')
plt.show()

def trainingDataSet(data, start, end):
    dataSet =[]
    for i in range(start, end):
        dataSet.append(data[i])
    return dataSet

# training data

x_trn = trainingDataSet(x_inp, 0, 150)
y_trn = trainingDataSet(y_out, 0, 150)

# test data

x_test = trainingDataSet(x_inp, 150, 200)
y_test = trainingDataSet(y_out, 150, 200)


def transformData(dataSet, d):
    return np.vander(dataSet, d, True)  # vandermonde matrix

def transpose(matrix):
    return matrix.transpose(1, 0)

# finding weigths
def trainModel(phi, label):

    phi_trans = transpose(phi)

    phi_conver = np.matmul(phi_trans, phi)

    phi_conver_inv = inv(phi_conver)

    a = np.matmul(phi_conver_inv, phi_trans)

    return np.matmul(a, label)

# mean sqare error
def testModel(w, phi, label):
    dataLen = len(label)
    error = 0
    for i in range(dataLen):
        error = error + np.square((label[i] - np.matmul(w.transpose(), phi[i])))
    return error/dataLen

e_mse = []
weight = []
i = 0
phi_test = []
for d in range(5,16):
    phi_trn =  transformData(x_trn, d+1)
    weight.append(trainModel(phi_trn, y_trn))
    phi_test.append(transformData(x_true, d+1))
    e_mse.append(testModel(weight[i], phi_test[i], y_test))
    i = i+1

y_degree = []
for k in range(11):
    y_prod = []
    for j in range(len(phi_test)):
        y_prod.append(np.matmul(weight[k].transpose(), phi_test[k].transpose()))
    y_degree.append(y_prod)
d= [5,6,7,8,9,10,11,12,13,14,15]
print(x_test)
print(y_degree[0][0])

d = [5,6,7,8,9,10,11,12,13,14,15,16]
t = PrettyTable(['d', 'e_mse'])
for i in range(0,12):
    t.add_row([str(d[i]), e_mse[i]])
print (t)

for i in range(11):
    for j in range(11):
        # x_new = np.linspace(x_test, y_degree[i][j], 50)
        if i == 0 : 
            color = 'b';    
        elif i == 1 :
            color = 'g';
        elif i == 2 :
            color = 'r';
        elif i == 3 :
            color = 'c';
        elif i == 4 :
            color = 'm';
        elif i == 5 :
            color = 'y';
        elif i == 6:
            color = 'k';
        elif i == 7 :
            color = 'steelblue';
        elif i == 8 :
            color = 'gold';
        elif i == 9 :
            color = 'orange';
        elif i == 10 :
            color = 'violet';                                                                                              
        
        plt.plot(x_true, y_degree[i][j],linestyle='-',linewidth=3.0, color= color, marker='.', markerfacecolor=color)


patch1 = mpatches.Patch(color='b', label='d= 5')
patch2 = mpatches.Patch(color='g', label='d= 6')
patch3 = mpatches.Patch(color='r', label='d= 7')
patch4 = mpatches.Patch(color='c', label='d= 8')
patch5 = mpatches.Patch(color='m', label='d= 9')
patch6 = mpatches.Patch(color='y', label='d= 10')
patch7 = mpatches.Patch(color='k', label='d= 11')
patch8 = mpatches.Patch(color='steelblue', label='d= 12')
patch9 = mpatches.Patch(color='gold', label='d= 12')
patch10 = mpatches.Patch(color='orange', label='d= 14')
patch11 = mpatches.Patch(color='violet', label='d= 15')

plt.xlabel('x_true')
plt.ylabel('y_transformed')
plt.legend(handles=[patch1,patch2,patch3,patch4,patch5,patch6,patch7,patch8,patch9,patch10,patch11])
# plt.plot(d, e_mse, linestyle='-', linewidth=3.0, color='g')


plt.show()
