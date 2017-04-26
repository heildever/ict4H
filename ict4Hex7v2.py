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
bad_cols = np.nonzero(data.sum(axis=0) == 0) # bad_cols corresponds to columns with 0 only
data = np.delete(data, bad_cols, axis=1)	# removing the bad_cols
# node numbers of hidden layers
lay1 = 64
lay2 = 32
# separating outputs
out = data[:,-1]
data = np.delete(data,-1,axis=1) ## removing the class data column from matrix
(rows, cols) = np.shape(data)
# manipulating the output --> converting it to a 452x16 matrix
zout = np.zeros((rows,16))
for i in range(0,rows-1):
	if int(out[i])!= 0 & int(out[i])!=16:
		zout[i,int(out[i])-1]=1
	elif int(out[i])==16:
		zout[i,-1]=1
		
# values for normalization
col_mean = np.mean(data, axis=0)
col_std = np.std(data, axis=0)		
# normalize the matrix
for i in range(0,cols):
        data[:,i] = (data[:,i]-col_mean[i])/col_std[i]		
        	
tf.set_random_seed(1234)
x = tf.placeholder(tf.float64,[rows,cols], name="input_tensor") # input tensor
t = tf.placeholder(tf.float64,[rows,16], name="output_tensor") # output tensor
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
w3 = tf.Variable(tf.random_normal([lay2,16], mean=0.0, stddev=1.0, dtype=tf.float64, name="weights3"))     
b3 = tf.Variable(tf.random_normal([rows,16], mean=0.0, stddev=1.0, dtype=tf.float64, name="biases3"))        
a3 = tf.matmul(z2,w3)+b3
y = tf.nn.softmax(a3) 
# output is a 452x16 matrix
# optimizer structure
cost=tf.reduce_sum(-tf.reduce_sum(t*tf.log(y)),name ="0bjectivefunction")
optim = tf.train.GradientDescentOptimizer(5e-4, name='Gradientdescent')
optim_op = optim.minimize(cost, var_list=[w1,b1,w2,b2,w3,b3])
# initialize
init = tf.initialize_all_variables()
# run the learning machine
sess = tf.Session()
sess.run(init)
# generate the data
xval = data
tval = zout
tval = np.reshape(tval,(rows,16))

for i in range(1000):
    # train 
    train_data = {x:xval, t:tval}
    sess.run(optim_op, feed_dict=train_data)
    if i % 500 == 0:
        print (i,cost.eval(feed_dict=train_data, session=sess))
# print the results        
# print(sess.run(w1), sess.run(b1))
# print(sess.run(w2), sess.run(b2))
# print(sess.run(w3), sess.run(b3))
# plot the results
result = np.zeros(rows) # to store output
yval=y.eval(feed_dict=train_data,session=sess)
yval = np.rint(yval)	# rounding the outputs to the nearest integer
yval = np.asarray(yval)
for i in range(rows):
	nonzero = np.nonzero(yval[i])
	print np.int_(nonzero)[0]
	result[i] = np.int_(nonzero)[0]
	
plt.plot(out,"ro", label="regressand")
plt.plot(yval,"bx", label="regression")
plt.xlabel("case number")
plt.grid(which="major", axis="both")
plt.legend()
#plt.savefig("one.pdf",format="pdf")
plt.show()
mse = ((out-result)**2).mean(axis=None)
print mse
