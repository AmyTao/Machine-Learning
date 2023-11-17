import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


# load dataset
data = pd.read_csv("/Users/tcj/Documents/GitHub/DDA3020/hw1/Regression.csv")
data = data.iloc[:, 2:]
# dealing with missing argument
data = data.dropna(how='any')
# 7590 remains
# separate features and target
X = data.drop(columns=['Next_Tmax', 'Next_Tmin'])
X = np.mat(X)
min_val = np.min(X,axis=0)
max_val = np.max(X,axis=0)
X = (X - min_val) / (max_val - min_val)

y = data[['Next_Tmax', 'Next_Tmin']]

# Set hyper parameters
epoch = 1000
random_seeds = [random.randint(1, 1000) for _ in range(10)]
lr_rate = 0.01
error_list = []

for i in range(2):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_seeds[i])
    local_error_list = []
    # initialize weights with rand number
    weight = np.random.rand(21, 2)
    # weight = np.random.rand(21, 2)
    # initialize bias with rand number
    bias = np.random.rand(1, 2)
    X_train = np.mat(X_train)
    # print(X_train.dtype)
    X_test = np.mat(X_test)
    y_train = np.mat(y_train)
    y_test = np.mat(y_test)
    # print(type(X_train))
    # print(X_train.shape)
    # print(X_train)
    # print(y_test)
    # print(y_train)


    for ep in range(epoch):
        pred = np.dot(X_train,weight)+bias
        train_error = np.sqrt((1/X_train.shape[0]) * np.sum(np.square(pred-y_train),axis = 0))
        # print(train_error)
        # local_error_list.append(train_error)
        # a=np.dot((pred-y_train).transpose(),X_train)
        gradient_w = (2/X_train.shape[0])*np.dot(X_train.transpose(),(pred-y_train))
        # print(gradient_w.shape)
        gradient_b = (2/X_train.shape[0])*np.sum((pred-y_train),axis = 0)
        weight -= gradient_w*lr_rate
        bias -= gradient_b*lr_rate
        if ep%100 == 0:
            local_error_list.append(np.squeeze(np.array(train_error)))
    x = [ i for i in range(0,epoch) if i%100 == 0]
    print(x)
    error1 = [i[0] for i in local_error_list]
    error2 = [i[1] for i in local_error_list]
    plt.plot(x, error1, label='train error of Next_Tmax', marker = 'o',color='blue',linestyle='-')
    plt.plot(x, error2, label='train error of Next_Tmin', marker = 'o',color='orange', linestyle='-')

    plt.xlabel('epoch')
    plt.ylabel('error')
    plt.legend()

    # Show the plot
    plt.grid(True)
    plt.show()



    pred = np.dot(X_test,weight)+bias
    test_error = np.sqrt((1/X_test.shape[0]) * np.sum(np.square(pred-y_test),axis = 0))
    # print("test_error")
    # print(pred[:10])
    # print(y_test[:10])
    print(test_error)

