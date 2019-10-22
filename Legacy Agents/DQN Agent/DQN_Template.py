# -*- coding: utf-8 -*-
"""
Created on Sun Oct  6 12:45:40 2019

@author: hydra li
"""

import numpy as np
from tqdm import tqdm

def Empty (*args):
    ''' An empty function which accepts any number of arguments '''
    pass

class DQN_Template:

    def __init__(self, Environment, Gamma, Epsilon_Range, Epsilon_Anneal, Model, Action_Discretise, Retrain_Frequency):
        '''
        Parameters
        ----------
            Environment       : The gym environment that the agent should train within.
            Action_Discretise : The discretisation of the action_space.
            Network_Params    : A dictionary or parameters used to Initialise the neural network used to approximate the quality function
                                which includes:
                                   Network Size  : A list of N ints where each entry represents the desired number of nodes in the Nth hidden layer
                                   Learning Rate : The learning rate used to update the weights and biases of the network
                                   Activation    : The activation function used. Acceptable inputs include ("Sigmoid", "Relu")
                                   Epoch         : The number of back propergation passes to be ran each time Fit is called.
                                   Alpha         : An L2 regularization term.
            Gamma             : The rate used to discount future reward when appraising the value of a state.
            Epsilon_Range     : The range of epsilon (for an epsilon greedy policy) across a training sequence
            Epsilon_Anneal    : The fraction of the training sequence which should have passed for epsilon to fall from its starting
                                value to its terminal value.
            Retrain_Frequency : The frequency at which the internal network is refitted (Measured in episode.)

        '''
        self.Environment       = Environment
        self.Gamma             = Gamma
        self.Epsilon_Range     = Epsilon_Range
        self.Epsilon_Anneal    = Epsilon_Anneal
        self.Retrain_Frequency = Retrain_Frequency
        self.Action_Dim        = Action_Discretise
        self.Action_Space      = np.linspace(Environment.action_space.low, Environment.action_space.high, Action_Discretise)
        self.Q_Network         = Model
        self.Exp               = []
        self.State_Dim         = Environment.observation_space.shape[0]
        self.Pool_Size         = 1e6

    def Train (self, N_Episodes,  Plot = Empty):
        '''
        Trains the agent

        Parameters
        ----------
            N_Episodes : The number of episodes to train the DQN Agent across.

        Notes
        -----
            Since the DQN learns off policy there should be no negative effects of calling Train() multiple times in series on
            the same agent. However perfered behaviour is to call this function only once as it ensures the agent slowly acts more
            optimally across the training sequence. (Epsilon will jump back to its inital value in subsequent calls to this function).
        '''

        epsilons = self.Epsilon_Range[0] * np.ones(N_Episodes) - (1 / self.Epsilon_Anneal) * np.arange(0, N_Episodes) * (self.Epsilon_Range[0] - self.Epsilon_Range[1])
        epsilons = np.maximum(epsilons, self.Epsilon_Range[1])
        Episode_Exps  = []

        for i in tqdm(range(N_Episodes)):

            State_0 = self.Environment.reset()
            Done = False
            Episode_Exp = []
            while Done == False:
                if np.random.uniform() > epsilons[i]:
                    Action_idx = np.random.choice(list(range(self.Action_Dim)))
                else:
                    Action_idx = self.Choose_Action(State_0)[0]

                State_1, Reward, Done, Info = self.Environment.step(self.Action_Space[Action_idx])
                Episode_Exp.append({"s0" : State_0, "s1" : State_1, "r" : Reward, "a" : Action_idx, "i" : Info, "done" : Done, 'Mu' :self.Action_Space[Action_idx]})
                State_0 = State_1

            self.Exp.extend(Episode_Exp)
            Episode_Exps.append(Episode_Exp)

            if len(self.Exp) > self.Pool_Size:
                self.Exp[0:len(Episode_Exp)] = []

            # Refit the model
            if i % self.Retrain_Frequency == 0:
                self.Refit()
                Plot(Episode_Exps)

    def Refit(self):
        ''' Refit the internal network. Accepts no arguments, since it uses the data stored in self.Exp '''
        pass

    def Predict_Q(self, state):
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
        pass

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
        pass

    def Execute_Action(self, state):
        '''
        Parameters
        ----------
            state | np.array (2D)
                A np array of a single state for which the action will be predicted.

        Returns
        -------
            float
                The actual action to take.
        '''

        return self.Action_Space[self.Choose_Action(state)]
