#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 27 22:03:11 2016

@author: heildever
"""
import scipy.io as scio
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

mat_file=scio.loadmat('parkinsonnew.mat')
data = mat_file.get('parkinsonnew')
# initial steps
(rows,cols) = np.shape(data) # 990x22
col_mean = np.mean(data, axis=0)
col_std = np.std(data, axis=0)
# normalize the matrix 
for i in range(cols):
        data[:,i] = (data[:,i]-col_mean[i])/col_std[i]    # data is a 990x17 matrix
        
out = data[:,6]
data = np.delete(data,[0,3,4,5,6],axis=1)
(rows,cols) = np.shape(data) # 990x17
tf.set_random_seed(1234)
x = tf.placeholder(tf.float64,[rows,cols], name="input_tensor") # input tensor
t = tf.placeholder(tf.float64,[rows,1], name="output_tensor") # output tensor
# neural network structure
w1 = tf.Variable(tf.random_normal([cols,1], mean=0.0, stddev=1.0, dtype=tf.float64, name="weights"))    # 
b1 = tf.Variable(tf.random_normal([rows,1], mean=0.0, stddev=1.0, dtype=tf.float64, name="biases"))    # 
y = tf.matmul(x,w1)+b1
# optimizer structure
cost = tf.reduce_sum(tf.squared_difference(y, t, name='objective_function'))
optim = tf.train.GradientDescentOptimizer(9e-5, name='Gradientdescent')
optim_op = optim.minimize(cost, var_list=[w1,b1])
# initialize
init = tf.initialize_all_variables()
# run the learning machine
sess = tf.Session()
sess.run(init)
# generate the data
xval = data
tval = out
tval = np.reshape(out, (rows,1))

for i in range(15000):
    # train 
    train_data = {x:xval, t:tval}
    sess.run(optim_op, feed_dict=train_data)
    if i % 2000 == 0:
        print (i,cost.eval(feed_dict=train_data, session=sess))
# print the results        
print(sess.run(w1), sess.run(b1))
    
# plot the results
yval=y.eval(feed_dict=train_data,session=sess)
plt.plot(tval,"ro", label="regressand")
plt.plot(yval,"bx", label="regression")
plt.xlabel("case number")
plt.grid(which="major", axis="both")
plt.legend()
#plt.savefig("one.pdf",format="pdf")
plt.show()
mse = ((tval-yval)**2).mean(axis=None)
print mse
