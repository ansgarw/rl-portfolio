import numpy as np

import tensorflow.compat.v1 as tf
tf.logging.set_verbosity(tf.logging.ERROR)

# import tensorflow as tf
# tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

from DQN_Template import DQN_Template


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

        # self.Weights = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
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



# The DQN Agent itself.
class DQN_MH(DQN_Template):

    def __init__ (self, Environment, Action_Discretise, Network_Params, Gamma = 0.98, Epsilon_Range = [1, 0.1], Epsilon_Anneal = 0.5, Retrain_Frequency = 25):

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
        Model = Multihead_TF(State_Dim,   Action_Discretise,
                            Hidden        = Network_Params["Network Size"],
                            Activation    = Network_Params["Activation"],
                            Learning_Rate = Network_Params["Learning Rate"],
                            Alpha         = Network_Params["Alpha"])

        super().__init__(Environment, Gamma, Epsilon_Range, Epsilon_Anneal, Model, Action_Discretise, Retrain_Frequency)

        self.Batch_Size        = Network_Params['Batch Size']
        self.Epoch             = Network_Params['Epoch']
        self.TF_Session        = tf.Session()
        self.TF_Session.run(tf.global_variables_initializer())

    def Refit(self):
        '''
        Refit the internal network.
        Accepts no arguments, since it uses the data stored in self.Exp
        '''

        if len(self.Exp) > self.Batch_Size*self.Epoch:
            Data = np.random.choice(self.Exp, size = (self.Epoch, self.Batch_Size), replace = False)
            for row in Data:
                X  = np.array([d['s0'] for d in row]).reshape((-1, self.State_Dim))
                Y  = np.array([d['r']  for d in row]).reshape((-1,1))
                S1 = np.array([d['s1'] for d in row]).reshape((-1, self.State_Dim))
                Done = np.array([(d['done'] == False) for d in row]).reshape(-1,1)

                # When generating the reward to fit towards we augment the immediate reward with the quality of the
                # subsequent state, assuming greedy action from that point on. This relationship is formalised by the
                # bellman equation. Q(s,a) = R + Argmax(a)(Q(s',a)) * Gamma
                S1_Val = self.Predict_Q(S1)*Done
                Y += S1_Val * self.Gamma
                self.TF_Session.run(self.Q_Network.fit, feed_dict={self.Q_Network.X: X, self.Q_Network.Q_In:Y})
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
        return self.TF_Session.run(self.Q_Network.Q, feed_dict={self.Q_Network.X: state})

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
        return self.TF_Session.run(self.Q_Network.choose, feed_dict={self.Q_Network.X: state})
