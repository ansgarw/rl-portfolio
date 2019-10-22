# -*- coding: utf-8 -*-
"""
Created on Sun Oct  6 12:56:39 2019

@author: hydra li
"""

import tensorflow as tf
import numpy as np

class Multihead_TF:
    def __init__(self, Input_Dim, Output_Dim = 1, Hidden = [20, 10], Activation = 'Sigmoid', Learning_Rate = 0.001, Alpha = 0):
        self._Activation_Method = self._Activation(Activation)
        regularizer = tf.contrib.layers.l2_regularizer(Alpha) if Alpha!= 0 else None
        self.X = tf.placeholder(shape = [None, Input_Dim], dtype = tf.float32, name = 'States')
        Hidden_Layers = self.X
        for layers in Hidden:
            Hidden_Layers = tf.layers.dense(Hidden_Layers, layers, activation= self._Activation_Method, activity_regularizer= regularizer)
        Q_Raw = tf.layers.dense(Hidden_Layers, Output_Dim, activation= None, activity_regularizer= regularizer)
        self.choose = tf.arg_max(Q_Raw, 1)
        self.Q = tf.reshape(tf.reduce_max(Q_Raw, 1), shape = (-1,1))

        self.Q_In = tf.placeholder(shape = [None, 1], dtype = tf.float32, name = 'Quality')
        self._Optimizer = tf.train.AdamOptimizer(learning_rate= Learning_Rate)
        self.loss = tf.losses.mean_squared_error(self.Q_In, self.Q)

#        self.Weights = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        self.fit = self._Optimizer.minimize(self.loss)

    def _Activation(self, how):
        if how == 'Sigmoid':
            A_fun = tf.nn.sigmoid
        elif how == 'Relu':
            A_fun = tf.nn.relu
        elif how == 'Tanh':
            A_fun = tf.nn.tanh
        elif how == 'Softplus':
            A_fun = tf.nn.softplus
        return A_fun


class Multihead_NP:
    def __init__ (self, Shape, Input_Dim, Output_Dim, Learning_Rate = 0.01, Epoch = 1, Activation = "Relu", Alpha = 0.005, Batch_Size = 200):
        '''
        Parameters
        ----------
            Shape         : A list of N ints in which the Nth int represents the number of hidden layers in the Nth layer
            Input_Dim     : The length of a single input observation (the number of nodes in the input layer)
            Output_Dim    : The length of a single output observation (the number of nodes in the optput layer)
            Learning_Rate : The rate at which the weights and biases of the network are updated.
            Epoch         : The number of backpropergation passes which are computed each time .Fit is called.
            Activation    : The activation function used by the hidden layers (output layer uses linear activation).
                            Acceptable agruments include ("Relu", "Sigmoid")
            Alpha         : An L2 regularization term, pass higher values to increase the penalisation applied to the network
                            for assinging high weights, and hopefully reduce overfitting.
            Batch_Size    : The size of a batch when applying SGD solver algorithm. Set to 0 to not use SGD.
        '''

        self.Weights = list()
        self.Biases  = list()
        self.Learning_Rate = Learning_Rate
        self.Output_Dim    = Output_Dim
        self.Epoch         = Epoch
        self.Alpha         = Alpha
        self.Batch_Size    = Batch_Size

        if Activation == "Relu":
            self.Act   = self.Relu
            self.d_Act = self.d_Relu
        elif Activation == "Sigmoid":
            self.Act   = self.Sigmoid
            self.d_Act = self.d_Sigmoid

        self.Shape = [Input_Dim] + Shape + [Output_Dim]
        for i in range(1, len(self.Shape)):
            self.Weights.append(np.random.normal(0, 1, (self.Shape[i-1], self.Shape[i])) / (self.Shape[i-1] + self.Shape[i]))
            self.Biases.append(np.random.normal(0, 1, (self.Shape[i], 1)))

        self.As      = [None] * (len(self.Weights) + 1)   #Pre Sigmoid
        self.Zs      = [None] * (len(self.Weights) + 1)   #Post Sigmoid

    def Sigmoid(self, X):
        ''' The sigmoid activation function '''
        return 1.0 / (1.0 + np.exp(-X))

    def d_Sigmoid(self, X):
        ''' The derivative of the sigmoid activation function '''
        return self.Sigmoid(X) * (1 - self.Sigmoid(X))

    def Relu(self, X):
        ''' The relu activation function '''
        return X * (X > 0)

    def d_Relu(self, X):
        ''' The drivative of the relu activation function '''
        return (X > 0)

    def d_Loss(self, Y_hat, Y):
        ''' The derivative of the Sum of Squared Error loss function '''
        return 2 * (Y_hat - Y)

    def Forward_Pass(self, X):
        '''
        Perform a forward pass, setting the pre and post activation values for all the nodes of the network.

        Parameters
        ----------
            X : A 2D numpy array of observations, where each row represents an observation.

        Notes
        -----
            This function should only be called internally by the class. Users attempting to genrate a
            prediction should call .Predict() instead.
        '''

        self.As = [None] * (len(self.Weights) + 1)
        self.Zs = [None] * (len(self.Weights) + 1)
        self.As[0] = X
        self.Zs[0] = X

        for i in range(1, len(self.As)):
            self.As[i] = np.matmul(self.Zs[i-1], self.Weights[i-1])

            for j in range(self.As[i].shape[0]):
                self.As[i][j] += self.Biases[i-1].reshape((-1))

            if i == len(self.As) - 1:
                self.Zs[i] = self.As[i]
            else:
                self.Zs[i] = self.Act(self.As[i])

    def BackProp(self, X_, Y_, Z_):
        '''
        Perform a back propergation pass and update the weights and biases of the network.
        The unique thing about this network is that the loss of each observation is used to train
        towards only one of the network heads, the loss for all other heads defaults to zero.

        Parameters
        ----------
            X_ : A 2D numpy array of observations, where each row represents an observation.
            Y_ : A 2D numpy array of output targets, where each row represents an observation.
            Z_ : A numpy column vector of ints in which the Nth entry represents the index of
                 the head of the network which should be trained for the Nth oberservation.

        Notes
        -----
            This function should only be called internally by the class. Users attempting to fit the
            network should call .Fit() instead.
        '''

        One_Hot = np.zeros((X_.shape[0], self.Output_Dim))
        Y       = np.zeros((X_.shape[0], self.Output_Dim))
        for i in range(X_.shape[0]):
            One_Hot[i][Z_[i]] = 1
            Y[i][Z_[i]]       = Y_[i]


        Grads = []
        for i in range(len(self.Weights))[::-1]:
            A      = self.As[i+1]
            Z      = self.Zs[i+1]
            Z_Prev = self.Zs[i]
            W      = self.Weights[i]

            dA = (self.d_Loss(Z, Y) * One_Hot).T if i+1 == len(self.Weights) else (self.d_Act(A).T * dZ)

            # get parameter gradients
            dW = np.matmul(dA, Z_Prev) / X_.shape[0] + ((self.Alpha * W.T) / X_.shape[0])
            dB = np.sum(dA, axis=1).reshape(-1,1) / X_.shape[0]
            Grads.append({'Weight' : dW, 'Bias' : dB})

            if i > 0:
                dZ = np.dot(W, dA)

        Grads = Grads[::-1]
        for i in range(len(Grads)):
            self.Weights[i] -= self.Learning_Rate * Grads[i]["Weight"].T
            self.Biases[i]  -= self.Learning_Rate * Grads[i]["Bias"]

    def Predict(self, X):
        '''
        Function to be called externally to generate a prediction from the network.

        Parameters
        ----------
            X : A 2D numpy array of observations, where each row represents an observation.

        Returns
        -------
            A 2D numpy array in which each row represents the output of each head for the ith oberservation
        '''

        self.Forward_Pass(X)
        return self.Zs[-1]

    def Fit(self, X, Y, Z):
        '''
        Function to be called externally to fit the network to a dataset.

        Parameters
        ----------
            X : A 2D numpy array of observations, where each row represents an observation.
            Y : A 2D numpy array of output targets, where each row represents an observation.
            Z : A numpy column vector of ints in which the Nth entry represents the index of
                the head of the network which should be trained for the Nth oberservation.
        '''

        if self.Batch_Size > 0:
            for _ in range(self.Epoch):
                idx = np.random.choice(X.shape[0], size = (X.shape[0] // self.Batch_Size, self.Batch_Size), replace = False)
                for i in range(idx.shape[0]):
                    self.Forward_Pass(X[idx[i]])
                    self.BackProp(X[idx[i]], Y[idx[i]], Z[idx[i]])

        else:
            for _ in range(self.Epoch):
                self.Forward_Pass(X)
                self.BackProp(X, Y, Z)
