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
            Environment | OpenAI Gym enviornment
                The environment to train the DQN within, ususally one of the Portfolio_Gym environments

            Gamma | float
                The rate used to discount future reward when appraising the value of a state.

            Epsilon_Range | list
                The range of epsilon (for an epsilon greedy policy) across a training sequence.

            Epsilon_Anneal | float
                The fraction of the training sequence which should have passed for epsilon to fall from its starting value to its terminal value.

            Model | N/A
                The neural network used by the agent. Can be numpy, tensorflow or scikit based.

            Action_Discretise | int
                The discretisation of the action_space.

            Retrain_Frequency | int
                The number of episodes between refits of the Network


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

    def Train (self, N_Episodes,  Plot = Empty, Diag = Empty):
        '''
        Trains the agent

        Parameters
        ----------
            N_Episodes | int
            The number of episodes the agent should be trained for. Note parameters like sigma and learning rate decay scale with the number of episodes.

            Plot | func
                A function pointer used by the Wrapper to plot the performance of the agent as it learns. This function is called every 10k steps.

            Diag | func
                A function pointer used by the wrapper to plot diagnostics (for example the sensitivity of Actor/Critic to state parameters). This function is called only 5 times throughout training.

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
                Episode_Exps = []

    def Refit(self):
        '''
        Refit the internal network.
        Accepts no arguments, since it uses the data stored in self.Exp
        '''
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
