import numpy                as np
import multiprocessing      as mp
import matplotlib.pyplot    as plt
from   mpl_toolkits.mplot3d import Axes3D

import copy
import math
import random
import pickle

from sklearn.neural_network import MLPRegressor

# CAHNGELOG - Version IV:
# Modified the input parameters to allow oversampling to be disabled.
# Fixed a bug whereby truncation error may lead to out of bounds actions.

# Deleted Value_Function(), as it was almost identical to Argmax_Action()

# Added Parameter to enable/disable training on recent experiance plus replay vs solely replay.
# Added Activation, Batch_Size and ArgMax_Steps as input params



class DQN_Agent():
    def __init__ (self, Env, Network_Size, Gamma = 0.99, Min_Exp = 100000, Min_Pos_Exp = 10000, Replay_Size = 50000, Update_Offline_Network = 25, Learning_Rate = 0.01, Over_Sample = False, Positive_Level = 0.0, Epoch = 1, Batch_Size = 200, Activation = 'logistic', ArgMax_Steps = 20, Train_Recent_Exp = True):

        self.Env = Env

        self.Action_Space = [[-1], [0], [1]]                  # Defines the actions this agent may take.
        self.Model_Space  = [None] * len(self.Action_Space)

        for i in range(len(self.Model_Space)):
            self.Model_Space[i] = MLPRegressor(hidden_layer_sizes = Network_Size,
                                               activation         = Activation,
                                               solver             = 'adam',
                                               warm_start         = True,
                                               max_iter           = Epoch,
                                               learning_rate_init = Learning_Rate,
                                               batch_size         = Batch_Size)

        self.Pos_Exp          = [[] for _ in range(len(self.Action_Space))]
        self.Positive_Level   = Positive_Level
        self.Over_Sample      = Over_Sample          # If Over_Sample == True, then  experiance replay will be 50% Pos_Exp
        self.Exp              = [[] for _ in range(len(self.Action_Space))]
        self.Train_Recent_Exp = Train_Recent_Exp

        self.Min_Exp     = Min_Exp
        self.Min_Pos_Exp = Min_Pos_Exp

        self.Update_Offline_Network = Update_Offline_Network

        self.Gamma = Gamma
        self.Replay_Size = Replay_Size
        self.ArgMax_Steps = ArgMax_Steps

        self.Model_Loaded = False
        self.Model_Iter_  = 0


    def Extend_Exp (self, Exp):
        for Ex in Exp:
            for i, Action in enumerate(self.Action_Space):
                if Action == Ex[1]:
                    self.Exp[i].append(Ex)
                    break


    def Extend_Pos_Exp (self, Exp):
        for Ex in Exp:
            for i, Action in enumerate(self.Action_Space):
                if Action == Ex[1]:
                    self.Pos_Exp[i].append(Ex)
                    break


    def Len_Pos_Exp (self):
        Size = 0
        for i in range(len(self.Action_Space)):
            Size += len(self.Pos_Exp[i])
        return Size


    def Len_Exp (self):
        Size = 0
        for i in range(len(self.Action_Space)):
            Size += len(self.Exp[i])
        return Size


    def Initialise_Experiance (self):
        # Gather some expereience by acting randomly and store it.

        i = 0
        while self.Len_Exp() < self.Min_Exp or (self.Len_Pos_Exp() < self.Min_Pos_Exp and self.Over_Sample == True) :
            # Sample some expereience
            State_0 = self.Env.reset()
            Done = False
            Episode_Exp = []
            i += 1

            while Done == False:
                Action = np.array(random.sample(self.Action_Space, 1))
                State_1, Reward, Done, Info = self.Env.step(Action)

                if Done == False:
                    Episode_Exp.append([list(State_0), list(Action), Reward, list(State_1)])
                    State_0 = copy.deepcopy(State_1)
                else:
                    Episode_Exp.append([list(State_0), list(Action), Reward, ([np.nan] * State_0.size)])
                    State_0 = copy.deepcopy(State_1)

                    if Reward > self.Positive_Level and self.Len_Pos_Exp() < self.Min_Pos_Exp and self.Over_Sample == True:
                        self.Extend_Pos_Exp(Episode_Exp)
                    elif self.Len_Exp() < self.Min_Exp:
                        self.Extend_Exp(Episode_Exp)

            if i % 100 == 0:
                print("Episode " + str(i) + "    Exp Pool : " + str(self.Len_Exp()) + "    Pos_Exp Pool : " + str(self.Len_Pos_Exp()))


        # next check if the model is new, or has been loaded from a previos runtime
        if self.Model_Loaded == False:
            # For the first step, train based upon immediate reward alone
            for i, Action in enumerate(self.Action_Space):
                if self.Over_Sample == True:
                    Data = self.Exp[i] + self.Pos_Exp[i]
                else:
                    Data = self.Exp[i]

                States  = np.array([x[0] for x in Data])
                Reward  = np.array([x[2] for x in Data])
                self.Model_Space[i].fit(States, Reward)


    def Argmax_Action (self, State):
        if np.isnan(State[0]):
            return random.sample(self.Action_Space, 1)[0], 0.0

        Qualities = []

        for Model in self.Model_Space:
            Qualities.append(Model.predict(State.reshape((1,-1))))

        Max_index = np.argmax(Qualities)

        return self.Action_Space[Max_index], Qualities[Max_index][0]


    def Train (self, N_Episodes, N_Fully_Random = 0, Epsilon_Override = None):
        # Check if the expereience pool has been initialised
        if len(self.Exp) < self.Min_Exp:
            self.Initialise_Experiance()

        Episode_Rewards   = []
        Episode_Epsilons  = []
        Recent_Exp        = []

        for i in range(N_Episodes):

            if Epsilon_Override == None:
                Epsilon = math.exp(-self.Model_Iter_ / 100) if (i >= N_Fully_Random) else 1.0
            else:
                Epsilon = Epsilon_Override

            State_0 = self.Env.reset()
            Done = False

            if i % self.Update_Offline_Network == 0:
                if len(Recent_Exp) > 0:
                    self.Train_Value_Function(Recent_Exp)
                    Recent_Exp = []


            Episode_Experiance = []
            while Done == False:

                if np.random.uniform(0,1) <= Epsilon:
                    Action = random.sample(self.Action_Space, 1)[0]
                else:
                    Action, _ = self.Argmax_Action(State_0)

                State_1, Reward, Done, Info = self.Env.step(Action)
                if Done == False:
                    Episode_Experiance.append([list(State_0), list(Action), Reward, list(State_1)])
                    Recent_Exp.append([list(State_0), list(Action), Reward, list(State_1)])
                    State_0 = copy.deepcopy(State_1)
                else:
                    Episode_Experiance.append([list(State_0), list(Action), Reward, ([np.nan] * State_0.size)])
                    Recent_Exp.append([list(State_0), list(Action), Reward, ([np.nan] * State_0.size)])
                    State_0 = copy.deepcopy(State_1)


            if Reward > self.Positive_Level and self.Over_Sample == True:
                self.Extend_Pos_Exp(Episode_Experiance)
            self.Extend_Exp(Episode_Experiance)

            Episode_Rewards.append(Reward)
            Episode_Epsilons.append(Epsilon)
            print(str(i), end = " ")

        return Episode_Rewards, Episode_Epsilons


    def Train_Value_Function (self, Recent_Exp):
        self.Model_Iter_ += 1

        if self.Over_Sample == True:
            if self.Train_Recent_Exp == True:
                for i, Action_ in enumerate(self.Action_Space):
                    N = int(min(self.Replay_Size / 2, len(self.Pos_Exp[i]), len(self.Exp[i])))
                    Data = random.sample(self.Exp[i], N)
                    Data.extend(random.sample(self.Pos_Exp[i], N))
                    Data.extend(Recent_Exp)

                    States        = np.array([x[0] for x in Data])
                    Reward        = np.array([x[2] for x in Data])
                    Future_States = np.array([x[3] for x in Data])

                    for j, State in enumerate(Future_States):
                        _ , Value = self.Argmax_Action(np.array(State))
                        Reward[j] += Value * self.Gamma

                    self.Model_Space[i].fit(States, Reward)


            else:
                for i, Action_ in enumerate(self.Action_Space):
                    N = int(min(self.Replay_Size / 2, len(self.Pos_Exp[i]), len(self.Exp[i])))
                    Data = random.sample(self.Exp[i], N)
                    Data.extend(random.sample(self.Pos_Exp[i], N))

                    States        = np.array([x[0] for x in Data])
                    Reward        = np.array([x[2] for x in Data])
                    Future_States = np.array([x[3] for x in Data])

                    for j, State in enumerate(Future_States):
                        _ , Value = self.Argmax_Action(np.array(State))
                        Reward[j] += Value * self.Gamma

                    self.Model_Space[i].fit(States, Reward)


        else:
            if self.Train_Recent_Exp == True:
                for i, Action_ in enumerate(self.Action_Space):
                    N = int(min(self.Replay_Size, len(self.Exp[i])))
                    Data = random.sample(self.Exp[i], N)
                    Data.extend(Recent_Exp)

                    States        = np.array([x[0] for x in Data])
                    Reward        = np.array([x[2] for x in Data])
                    Future_States = np.array([x[3] for x in Data])

                    for j, State in enumerate(Future_States):
                        _ , Value = self.Argmax_Action(np.array(State))
                        Reward[j] += Value * self.Gamma

                    self.Model_Space[i].fit(States, Reward)


            else:
                for i, Action_ in enumerate(self.Action_Space):
                    N = int(min(self.Replay_Size, len(self.Exp[i])))
                    Data = random.sample(self.Exp[i], N)

                    States        = np.array([x[0] for x in Data])
                    Reward        = np.array([x[2] for x in Data])
                    Future_States = np.array([x[3] for x in Data])

                    for j, State in enumerate(Future_States):
                        _ , Value = self.Argmax_Action(np.array(State))
                        Reward[j] += Value * self.Gamma

                    self.Model_Space[i].fit(States, Reward)







    # Auxillary Functions
    def Print_V_Function (self):
        V_Function = []

        dx = (self.Env.observation_space.high[0] - self.Env.observation_space.low[0]) / 50
        dy = (self.Env.observation_space.high[1] - self.Env.observation_space.low[1]) / 50

        x_low = self.Env.observation_space.low[0]
        y_low = self.Env.observation_space.low[1]

        for i in range(50):
            for j in range(50):
                State = [x_low + dx * i, y_low + dy * j]
                _, Value = self.Argmax_Action(np.array(State))
                V_Function.append([x_low + i * dx, y_low + j * dy, Value])
        V_Function = np.array(V_Function)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(V_Function[:,0], V_Function[:,1], V_Function[:,2], zdir='z', c= 'red')
        plt.show()


    def Print_Action_Space (self):
        V_Function = []

        dx = (self.Env.observation_space.high[0] - self.Env.observation_space.low[0]) / 50
        dy = (self.Env.observation_space.high[1] - self.Env.observation_space.low[1]) / 50

        x_low = self.Env.observation_space.low[0]
        y_low = self.Env.observation_space.low[1]

        for i in range(50):
            for j in range(50):
                State = [x_low + dx * i, y_low + dy * j]
                Action, _ = self.Argmax_Action(np.array(State))
                V_Function.append([x_low + i * dx, y_low + j * dy, Action[0]])
        V_Function = np.array(V_Function)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(V_Function[:,0], V_Function[:,1], V_Function[:,2], zdir='z', c= 'red')
        plt.show()


    def Decide_Action (self, State):
        Action, _ = self.Argmax_Action(np.array(State))
        return Action


    def Save (self, FileName):
        with open(FileName, 'wb') as file:
            pickle.dump([self.Model_Space, self.Model_Iter_], file)


    def Load (self, FileName):
        with open(FileName, 'rb') as file:
            Loaded_Data = pickle.load(file)
            self.Model_Space = Loaded_Data[0]
            self.Model_Iter_  = Loaded_Data[1]
        self.Model_Loaded = True



















# End
