import numpy             as np
import matplotlib.pyplot as plt
import tqdm


def Empty (*args):
    ''' An empty function which accepts any number of arguments '''
    pass


class A2C_Template:

    def __init__ (self, Environment, Gamma, Retrain_Frequency):

        self.Environment       = Environment
        self.Gamma             = Gamma
        self.Retrain_Frequency = Retrain_Frequency

        self.State_Dim         = Environment.observation_space.shape[0]
        self.Action_Dim        = Environment.action_space.shape[0]


    def Train (self, N_Episodes, Plot = Empty):
        '''
        Train the AC agent in its specified environment.

        Parameters
        ----------
            N_Episodes : The number of episodes the agent should be trained for. Note parameters like
                         sigma and learning rate decay scale with the number of episodes.

            Plot       : A list of plots which the function should return.
                         Accepted inputs include:
        '''
        
        self.N_Updates = N_Episodes/self.Retrain_Frequency
        Exp = []
#        Episode_Exps = []
        Step_Count = 0

        for i in tqdm.tqdm(range(N_Episodes)):
            Done = False
            Episode_Exp = []
            State_0 = self.Environment.reset()

            while Done == False:
                Mu, Sigma = self.Predict_Action(State_0.reshape(1, self.State_Dim))
                Action = np.random.normal(Mu, Sigma).flatten()

                State_1, Reward, Done, Info = self.Environment.step(Action)
                Episode_Exp.append({"s0" : State_0, "s1" : State_1, "r" : Reward, "a" : Action,'done':Done, 'i' : Info, 'Mu' : Mu.flatten(), 'Sigma' : Sigma.flatten()})
                State_0 = State_1
                Step_Count += 1


            Episode_Exp = self.BackProp_Reward(Episode_Exp)
#            Exp.extend(Episode_Exp)
            Exp.append(Episode_Exp)

            if Step_Count > 10000:
                Plot(Exp)
                Exp = []
                Step_Count = 0

            if i % self.Retrain_Frequency == 0:
                self.Refit_Model(Exp)
                Exp = []


    def Predict_Action (self, State):
        pass


    def BackProp_Reward (self, Episode_Exp):
        ''' The default is to use standard MC method (May be overloaded to allow for GAE) '''

        for j in range(len(Episode_Exp) - 1)[::-1]:
            Episode_Exp[j]["r"] += Episode_Exp[j+1]["r"] * self.Gamma

        return Episode_Exp


    def Refit_Model (self, Exp):
        pass
