#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 30 18:52:16 2016

@author: heildever
"""
import scipy.io as scio
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

mat_file=scio.loadmat('parkinsonnew.mat')
data = mat_file.get('parkinsonnew')

# node numbers of hidden layers
lay1 = 18
lay2 = 10
(rows,cols) = np.shape(data) # 990x17
col_mean = np.mean(data, axis=0)
col_std = np.std(data, axis=0)
# normalize the matrix 
for i in range(cols):
        data[:,i] = (data[:,i]-col_mean[i])/col_std[i]    # data is a 990x17 matrix
# separating the output column      
out = data[:,4]
data = np.delete(data,[0,3,4,5,6],axis=1)
(rows,cols) = np.shape(data) # 990x17
# normalize the matrix 
for i in range(cols):
        data[i] = (data[i]-col_mean[i])/col_std[i] 
      
tf.set_random_seed(1234)
x = tf.placeholder(tf.float64,[rows,cols], name="input_tensor") # input tensor
t = tf.placeholder(tf.float64,[rows,1], name="output_tensor") # output tensor
# neural network structure
w1 = tf.Variable(tf.random_normal([cols,lay1], mean=0.0, stddev=1.0, dtype=tf.float64, name="weights1"))   
b1 = tf.Variable(tf.random_normal([rows,lay1], mean=0.0, stddev=1.0, dtype=tf.float64, name="biases1"))    
a1 = tf.matmul(x,w1)+b1
z1 = tf.nn.sigmoid(a1)
# first layer 
w2 = tf.Variable(tf.random_normal([lay1,lay2], mean=0.0, stddev=1.0, dtype=tf.float64, name="weights2"))    
b2 = tf.Variable(tf.random_normal([rows,lay2], mean=0.0, stddev=1.0, dtype=tf.float64, name="biases2"))             
a2 = tf.matmul(z1,w2)+b2
z2 = tf.nn.sigmoid(a2)
# second layer
w3 = tf.Variable(tf.random_normal([lay2,1], mean=0.0, stddev=1.0, dtype=tf.float64, name="weights3"))     
b3 = tf.Variable(tf.random_normal([rows,1], mean=0.0, stddev=1.0, dtype=tf.float64, name="biases3"))        
y = tf.matmul(z2,w3)+b3
# optimizer structure
cost = tf.reduce_sum(tf.squared_difference(y, t, name='objective_function'))
optim = tf.train.GradientDescentOptimizer(2e-4, name='Gradientdescent')
optim_op = optim.minimize(cost, var_list=[w1,b1,w2,b2,w3,b3])
# initialize
init = tf.initialize_all_variables()
# run the learning machine
sess = tf.Session()
sess.run(init)
# generate the data
xval = data
tval = out
tval = np.reshape(tval,(rows,1))

for i in range(7000):
    # train 
    train_data = {x:xval, t:tval}
    sess.run(optim_op, feed_dict=train_data)
    if i % 1000 == 0:
        print (i,cost.eval(feed_dict=train_data, session=sess))
# print the results        
# print(sess.run(w1), sess.run(b1))
# print(sess.run(w2), sess.run(b2))
# print(sess.run(w3), sess.run(b3))
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

