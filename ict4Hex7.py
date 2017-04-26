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

mat_file=scio.loadmat('arrhythmia.mat')
data = mat_file.get('arrhythmia')
# removing columns with 0's only
bad_cols = np.nonzero(data.sum(axis=0) == 0)
data = np.delete(data, bad_cols, axis=1)
# separating outputs
out = np.clip(data[:,-1],0,2) # filtering the output as 1's and 2's
for i in range(len(out)-1):
	if out[i]==1:
		out[i]=0
	else:
		out[i]=1	

data = np.delete(data,-1,axis=1) ## removing the class data column from matrix
(rows, cols) = np.shape(data)
# values for normalization
col_mean = np.mean(data, axis=0)
col_std = np.std(data, axis=0)
# node numbers of hidden layers
lay1 = cols
lay2 = 128
# normalize the matrix
for i in range(0,cols):
        data[:,i] = (data[:,i]-col_mean[i])/col_std[i]
        
tf.set_random_seed(1234)
x = tf.placeholder(tf.float64,[rows,cols], name="input_tensor") # input tensor
t = tf.placeholder(tf.float64,[rows,1], name="output_tensor") # output tensor
# neural network structure
# first layer 
w1 = tf.Variable(tf.random_normal([cols,lay1], mean=0.0, stddev=1.0, dtype=tf.float64, name="weights1"))   
b1 = tf.Variable(tf.random_normal([rows,lay1], mean=0.0, stddev=1.0, dtype=tf.float64, name="biases1"))    
a1 = tf.matmul(x,w1)+b1
z1 = tf.nn.sigmoid(a1)
# second layer
w2 = tf.Variable(tf.random_normal([lay1,lay2], mean=0.0, stddev=1.0, dtype=tf.float64, name="weights2"))    
b2 = tf.Variable(tf.random_normal([rows,lay2], mean=0.0, stddev=1.0, dtype=tf.float64, name="biases2"))             
a2 = tf.matmul(z1,w2)+b2
z2 = tf.nn.sigmoid(a2)
# output layer
w3 = tf.Variable(tf.random_normal([lay2,1], mean=0.0, stddev=1.0, dtype=tf.float64, name="weights3"))     
b3 = tf.Variable(tf.random_normal([rows,1], mean=0.0, stddev=1.0, dtype=tf.float64, name="biases3"))        
a3 = tf.matmul(z2,w3)+b3
y = tf.nn.sigmoid(a3)
# optimizer structure
cost = tf.reduce_sum(tf.squared_difference(y, t, name='objective_function'))
optim = tf.train.GradientDescentOptimizer(99e-4, name='Gradientdescent')
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

for i in range(1000):
    # train 
    train_data = {x:xval, t:tval}
    sess.run(optim_op, feed_dict=train_data)
    if i % 100 == 0:
        print (i,cost.eval(feed_dict=train_data, session=sess))
# print the results        
# print(sess.run(w1), sess.run(b1))
# print(sess.run(w2), sess.run(b2))
# print(sess.run(w3), sess.run(b3))
# plot the results
yval=y.eval(feed_dict=train_data,session=sess)
yval = np.rint(yval)	# rounding the outputs to the nearest integer
plt.plot(tval,"ro", label="regressand")
plt.plot(yval,"bx", label="regression")
plt.xlabel("case number")
plt.grid(which="major", axis="both")
plt.legend()
#plt.savefig("one.pdf",format="pdf")
plt.tight_layout()
plt.show()
mse = ((tval-yval)**2).mean(axis=None)
print mse
