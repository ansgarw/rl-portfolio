# -*- coding: utf-8 -*-
"""
Created on Sun Oct  6 18:01:44 2019

@author: hydra li
"""

import numpy as np
from Neural_Network_Collections import Multihead_NP
from DQN_Template import DQN_Template

class DQN_MH_NP(DQN_Template):
    
    def __init__ (self, Environment, Action_Discretise, Network_Params, Gamma = 0.99, Epsilon_Range = [1, 0.1], Epsilon_Anneal = 0.5, Retrain_Frequency = 25):

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
        state = state.reshape(-1, self.State_Dim)
        ''' A simple pedict fucntion to extract predictions from the agent without having to call methods on the internal network '''
        return np.amax(self.Q_Network.Predict(state), axis = 1)

    def Choose_Action(self, state):
        state = state.reshape(-1, self.State_Dim)
        ''' Returns the optimal action per the Q Network '''
        return np.argmax(self.Q_Network.Predict(state), axis = 1)




