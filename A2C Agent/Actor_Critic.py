# The definitive actor critic
# Powered by tensorflow
# Utilising multiple networks for policy and value function

import numpy      as np
import tensorflow as tf
import warnings

from A2C_Template import A2C_Template

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)



class Actor_Network :

    def __init__ (self, Shape, Input_Dim, Output_Dim, Learning_Rate = 0.01, Epoch = 1, Activation = "Relu", Solver = 'ADAM', Alpha = 0.005, Batch_Size = 200):

        self.Learning_Rate = Learning_Rate
        self.Epoch         = Epoch
        self.Batch_Size    = Batch_Size
        self.Input_Dim     = Input_Dim
        self.Output_Dim    = Output_Dim

        # Create placeholders for the input parameters
        self.State     = tf.placeholder(shape = [None, Input_Dim],  dtype = tf.float32, name = 'State')
        self.Action    = tf.placeholder(shape = [None, Output_Dim], dtype = tf.float32, name = 'Action')
        self.Sigma     = tf.placeholder(shape = [None, Output_Dim], dtype = tf.float32, name = 'Sigma')
        self.Advantage = tf.placeholder(shape = [None, 1], dtype = tf.float32, name = 'Advantage')

        self.Learning_Rate = tf.placeholder(shape = (), dtype = tf.float32, name = 'Learning_rate')
        self.Default_LR    = Learning_Rate

        self.Activation_Method = self.Select_Activation(Activation)
        self.Optimiser         = self.Select_Optimiser(Solver)

        Alpha = tf.contrib.layers.l2_regularizer(Alpha) if Alpha != 0 else None

        # Next create the hidden layers
        Last_Layer = self.State
        for Layer in Shape:
            Last_Layer = tf.layers.dense(Last_Layer, Layer, activation = self.Activation_Method, activity_regularizer = Alpha)

        # Create the output layer
        self.Mu = tf.layers.dense(Last_Layer, Output_Dim, activation = None, activity_regularizer = Alpha)

        Loglik = (tf.log(2 * np.pi * self.Sigma ** 2) / 2) + (0.5 * (self.Action - self.Mu) ** 2 / (self.Sigma ** 2))
        Loss   = tf.matmul(tf.reshape(tf.reduce_sum(-Loglik, axis = 1), shape = [1,-1]), self.Advantage)

        self._Fit = self.Optimiser.minimize(Loss)

        self.TF_Session = tf.Session()
        self.TF_Session.run(tf.global_variables_initializer())


    def Select_Activation (self, Method):
        ''' Specify the activation function from its keyword. '''

        if Method == 'Sigmoid':
            return tf.nn.sigmoid
        elif Method == 'Relu':
            return tf.nn.relu
        elif Method == 'Tanh':
            return tf.nn.tanh
        elif Method == 'Softplus':
            return tf.nn.Softplus
        else:
            warnings.warn('Activation function ' + str(Method) + ' not recognised, defaulting to Sigmoid')
            return tf.nn.sigmoid


    def Select_Optimiser (self, Method):
        ''' Specify the solver from its keyword. '''

        if Method == 'ADAM':
            return tf.train.AdamOptimizer(learning_rate = self.Learning_Rate)
        elif Method == 'SGD':
            return tf.train.GradientDescentOptimizer(learning_rate = self.Learning_Rate)

        else:
            warnings.warn('Optimiser keyword ' + str(Method) + ' not recognised, defaulting to ADAM')
            return tf.train.AdamOptimizer(learning_rate = self.Learning_Rate)


    def Predict (self, State):
        return self.TF_Session.run(self.Mu, feed_dict = {self.State: State.reshape(-1, self.Input_Dim)})


    def Fit (self, State, Action, Advantage, Sigma, Learning_Rate = 0):
        if Learning_Rate == 0 : Learning_Rate = self.Default_LR
        self.TF_Session.run(self._Fit, feed_dict = {self.State         : State,
                                                    self.Action        : Action,
                                                    self.Advantage     : Advantage,
                                                    self.Sigma         : Sigma,
                                                    self.Learning_Rate : Learning_Rate})



class Critic_Network :


    def __init__ (self, Shape, Input_Dim, Output_Dim, Learning_Rate = 0.01, Epoch = 1, Activation = "Relu", Solver = 'ADAM', Alpha = 0.005, Batch_Size = 200):

        self.Learning_Rate = Learning_Rate
        self.Epoch         = Epoch
        self.Batch_Size    = Batch_Size
        self.Input_Dim     = Input_Dim
        self.Output_Dim    = Output_Dim

        # Create placeholders for the input parameters
        self.State = tf.placeholder(shape = [None, Input_Dim],  dtype = tf.float32, name = 'State')
        self.Value = tf.placeholder(shape = [None, Output_Dim],  dtype = tf.float32, name = 'Value')

        self.Activation_Method = self.Select_Activation(Activation)
        self.Optimiser         = self.Select_Optimiser(Solver)

        Alpha = tf.contrib.layers.l2_regularizer(Alpha) if Alpha != 0 else None

        # Next create the hidden layers
        Last_Layer = self.State
        for Layer in Shape:
            Last_Layer = tf.layers.dense(Last_Layer, Layer, activation = self.Activation_Method, activity_regularizer = Alpha)

        # Create the output layer
        self.Value_Prediction = tf.layers.dense(Last_Layer, Output_Dim, activation = None, activity_regularizer = Alpha)

        Loss = tf.losses.mean_squared_error(self.Value, self.Value_Prediction)

        self._Fit = self.Optimiser.minimize(Loss)

        self.TF_Session = tf.Session()
        self.TF_Session.run(tf.global_variables_initializer())


    def Select_Activation (self, Method):
        ''' Specify the activation function from its keyword. '''

        if Method == 'Sigmoid':
            return tf.nn.sigmoid
        elif Method == 'Relu':
            return tf.nn.relu
        elif Method == 'Tanh':
            return tf.nn.tanh
        elif Method == 'Softplus':
            return tf.nn.Softplus
        else:
            warnings.warn('Activation function ' + str(Method) + ' not recognised, defaulting to Sigmoid')
            return tf.nn.sigmoid


    def Select_Optimiser (self, Method):
        ''' Specify the solver from its keyword. '''

        if Method == 'ADAM':
            return tf.train.AdamOptimizer(learning_rate = self.Learning_Rate)
        elif Method == 'SGD':
            return tf.train.GradientDescentOptimizer(learning_rate = self.Learning_Rate)

        else:
            warnings.warn('Optimiser keyword ' + str(Method) + ' not recognised, defaulting to ADAM')
            return tf.train.AdamOptimizer(learning_rate = self.Learning_Rate)


    def Predict (self, State):
        return self.TF_Session.run(self.Value_Prediction, feed_dict = {self.State: State.reshape(-1, self.Input_Dim)})


    def Fit (self, State, Value):
        self.TF_Session.run(self._Fit, feed_dict = {self.State : State,
                                                    self.Value : Value})




class Actor_Critic (A2C_Template):

    def __init__ (self, Environment, Actor_Params, Critic_Params, Gamma, Sigma_Range, Sigma_Anneal, Retrain_Frequency):

        super().__init__(Environment, Gamma, Retrain_Frequency)

        self.Sigma_Range   = Sigma_Range
        self.Sigma_Anneal  = Sigma_Anneal
        self.Actor_Hypers  = Actor_Params
        self.Critic_Hypers = Critic_Params

        self.Sigma = self.Sigma_Range[0]

        self.Actor = Actor_Network(self.Actor_Hypers["Network Size"], self.State_Dim, self.Action_Dim,
                                   Learning_Rate = self.Actor_Hypers["Learning Rate"],
                                   Activation    = self.Actor_Hypers["Activation"],
                                   Solver        = self.Actor_Hypers['Solver'],
                                   Epoch         = self.Actor_Hypers["Epoch"],
                                   Alpha         = self.Actor_Hypers["Alpha"],
                                   Batch_Size    = self.Actor_Hypers["Batch Size"])

        self.Critic = Critic_Network(self.Critic_Hypers["Network Size"], self.State_Dim, 1,
                                     Learning_Rate = self.Critic_Hypers["Learning Rate"],
                                     Activation    = self.Critic_Hypers["Activation"],
                                     Solver        = self.Critic_Hypers['Solver'],
                                     Epoch         = self.Critic_Hypers["Epoch"],
                                     Alpha         = self.Critic_Hypers["Alpha"],
                                     Batch_Size    = self.Critic_Hypers["Batch Size"])


    def Train (self, N_Episodes, *args):

        self.Sigma_Fit_Anneal = self.Sigma_Anneal * int(N_Episodes / self.Retrain_Frequency)
        self.Refit_Count = 0

        if len(args) == 1:
            super().Train(N_Episodes, args[0])
        else:
            super().Train(N_Episodes)


    def Predict_Action (self, State):
        return self.Actor.Predict(State), np.array([self.Sigma])


    def Refit_Model (self, Exp):
        ''' Refit the network. '''

        State     = np.array([e['s0'] for e in Exp]).reshape((-1, self.State_Dim))
        Action    = np.array([e['a']  for e in Exp]).reshape((-1, self.Action_Dim))
        Reward    = np.array([e['r']  for e in Exp]).reshape((-1, 1))
        Advantage = Reward - self.Critic.Predict(State)

        Learning_Rate = self.Actor_Hypers['Learning Rate'] * self.Sigma
        self.Actor.Fit(State, Action, Advantage, np.array([self.Sigma]).reshape(1,1), Learning_Rate)
        self.Critic.Fit(State, Reward)

        self.Refit_Count += 1
        self.Sigma = max(self.Sigma_Range[0] - ((self.Sigma_Range[0] - self.Sigma_Range[1]) * (self.Refit_Count / self.Sigma_Anneal)), self.Sigma_Range[1])
