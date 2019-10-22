# -*- coding: utf-8 -*-
"""
Created on Sun Oct  6 18:01:44 2019

@author: hydra li
"""

import numpy as np
from DQN_Template import DQN_Template

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

        for _ in range(self.Epoch):

            if self.Batch_Size == 0:
                self.Forward_Pass(X)
                self.BackProp(X, Y, Z)

            else:
                idx = np.random.choice(X.shape[0], size = (X.shape[0] // self.Batch_Size, self.Batch_Size), replace = False)
                for i in range(idx.shape[0]):
                    self.Forward_Pass(X[idx[i]])
                    self.BackProp(X[idx[i]], Y[idx[i]], Z[idx[i]])



class DQN_MH_NP(DQN_Template):

    def __init__ (self, Environment, Action_Discretise, Network_Params, Gamma = 0.99, Epsilon_Range = [1, 0.1], Epsilon_Anneal = 0.5, Retrain_Frequency = 25):

        '''
        Parameters
        ----------
            Environment | OpenAI Gym enviornment
                The environment to train the DQN within, ususally one of the Portfolio_Gym environments

            Gamma | float
                The rate used to discount future reward when appraising the value of a state.

            Epsilon_Range | list
                The range of epsilon (for an epsilon greedy policy) across a training sequence.

            Epsilon_Anneal | float
                The fraction of the training sequence which should have passed for epsilon to fall from its starting value to its terminal value.

            Action_Discretise | int
                The discretisation of the action_space.

            Retrain_Frequency | int
                The number of episodes between refits of the Network

            Network_Params | dict
                'Network Size'  | int         | The size of the Actor NN
                'Learning Rate' | float, list | The learning rate of the Actor. list for cyclical learning rate, see network annotation.
                'Activation'    | string      | The activation of the Actor. Acceptable inputs include ['Relu', 'Sigmoid']
                'Epoch'         | int         | The Actor Epoch
                'Alpha'         | float       | L2 regularization coefficient for the Actor.
                'Batch_Size'    | int         | The size of the minibatch if enabling SGD in the network. Set to zero to disable SGD

        '''

        State_Dim = Environment.observation_space.shape[0]
        Model = Multihead_NP(Network_Params["Network Size"], State_Dim, Action_Discretise,
                               Learning_Rate = Network_Params["Learning Rate"],
                               Activation    = Network_Params["Activation"],
                               Epoch         = Network_Params["Epoch"],
                               Alpha         = Network_Params["Alpha"],
                               Batch_Size    = Network_Params["Batch_Size"])

        self.Sample_size = Network_Params['Epoch'] * Network_Params['Batch_Size']
        super().__init__(Environment, Gamma, Epsilon_Range, Epsilon_Anneal, Model, Action_Discretise, Retrain_Frequency)

    def Refit(self):
        '''
        Refit the internal network.
        Accepts no arguments, since it uses the data stored in self.Exp
        '''

        if len(self.Exp) > self.Sample_size:
            Data = np.random.choice(self.Exp, size = self.Sample_size, replace = False)
            X  = np.array([d['s0'] for d in Data]).reshape((-1, self.State_Dim))
            Y  = np.array([d['r']  for d in Data])
            Z  = np.array([d['a']  for d in Data]).reshape((-1, 1))
            S1 = np.array([d['s1'] for d in Data]).reshape((-1, self.State_Dim))

            # When generating the reward to fit towards we augment the immediate reward with the quality of the
            # subsequent state, assuming greedy action from that point on. This relationship is formalised by the
            # bellman equation. Q(s,a) = R + Argmax(a)(Q(s',a)) * Gamma
            S1_Val = self.Q_Network.Predict(S1)
            S1_Val = np.amax(S1_Val, axis = 1) * np.array([(d['done'] == False) for d in Data])
            Y = (Y + S1_Val * self.Gamma).reshape(-1, 1)

            self.Q_Network.Fit(X, Y, Z)
        else:
            pass

    def Predict_Q (self, state):
        '''
        Parameters
        ----------
            state | np.array (2D)
                A np array of states or a single state for which the value will be predicted.

        Returns
        -------
            np.array (1D)
                A 1D numpy array of qualities for each state passed.
        '''

        state = state.reshape(-1, self.State_Dim)
        return np.amax(self.Q_Network.Predict(state), axis = 1)

    def Choose_Action(self, state):
        '''
        Parameters
        ----------
            state | np.array (2D)
                A np array of states or a single state for which the action will be predicted.

        Returns
        -------
            np.array (1D)
                A 1D numpy array of actions indexes for each state passed.
        '''

        state = state.reshape(-1, self.State_Dim)
        return np.argmax(self.Q_Network.Predict(state), axis = 1)
