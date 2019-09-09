import numpy                as np
import multiprocessing      as mp
import matplotlib.pyplot    as plt
from   mpl_toolkits.mplot3d import Axes3D

import copy
import math
import random
import pickle
import gym
# import gym_Merton

from sklearn.neural_network import MLPRegressor

# CHANGELOG - Version IV:
# Modified the input parameters to allow oversampling to be disabled.
# Fixed a bug whereby truncation error may lead to out of bounds actions.

# Deleted Value_Function(), as it was almost identical to Argmax_Action()

# Added Parameter to enable/disable training on recent experiance plus replay vs solely replay.
# Added Activation, Batch_Size and ArgMax_Steps as input params

# CHANGELOG - Version V:
# Added asynchronous initialisation



class DQN_Agent():
    def __init__ (self, Env_String, Network_Size, Gamma = 0.99, Min_Exp = 100000, Min_Pos_Exp = 10000, Replay_Size = 50000, Update_Network = 25, Learning_Rate = 0.01, Over_Sample = False, Positive_Level = 0.0, Epoch = 1, Batch_Size = 200, Activation = 'logistic', ArgMax_Steps = 20, Train_Recent_Exp = True):

        # Since this version is asynchronous it must create multiple copies of the environment,
        # hence it requires the Envs string.
        self.Env_String = Env_String
        self.Env = gym.make(self.Env_String)


        # The asynchronous workload is as follows:
        # N workers complete M episodes each before collecting the results together for processing, these
        # two parameters are set in Init_Async_Settings = [M, N].
        # The workers cannot change any parameters internal to the class themselves, only record data.
        # So far only initialisation is asynchronous, as the bottleneck in TD training is the neural_network fitting.
        self.Init_Async_Settings   = [200, 4]

        self.Online_Model  = MLPRegressor(hidden_layer_sizes = Network_Size,
                                          activation         = Activation,
                                          solver             = 'adam',
                                          warm_start         = True,
                                          max_iter           = Epoch,
                                          learning_rate_init = Learning_Rate,
                                          batch_size         = Batch_Size)

        # Exp and Pos_Exp are the experiance replay pools, Pos_Exp is only used if Over_Sample == True,
        # in which case positive experiances are recorded seperatly and oversampled in the training process,
        # allowing the network to converge in environments where positive experiances are rare.
        # Positive_Level is the required value for a terminal reward to be considered positive.
        # Train_Recent_Exp is a bool indicating whether recent expericance should be included in the
        # training process, or just added to the experiance replay.
        self.Pos_Exp          = []
        self.Positive_Level   = Positive_Level
        self.Over_Sample      = Over_Sample          # If Over_Sample == True, then  experiance replay will be 50% Pos_Exp
        self.Exp              = []
        self.Train_Recent_Exp = Train_Recent_Exp

        # Min experiance levels before TD training begins.
        self.Min_Exp     = Min_Exp
        self.Min_Pos_Exp = Min_Pos_Exp

        # Frequency with which the network is updated
        self.Update_Network = Update_Network

        # Gamma is rate of decay of reward when propergated backwards through preceding states.
        # Replay size gives the number of replay steps to include in each training process
        # ArgMax_Steps is the decretisation of the action space when choosing an optimal action
        # for a given state. See the argmax function for a detailed description of its use.
        self.Gamma = Gamma
        self.Replay_Size = Replay_Size
        self.ArgMax_Steps = ArgMax_Steps

        # Model loaded is a flag to indicate whether the network has been loaded from file,
        # and hence does not require re-initialisation.
        # Model_iter counts the number of times the network is trained, and is used to set Epsilon
        # (of an Epsilon greedy policy)
        self.Model_Loaded = False
        self.Model_Iter_  = 0



    def Initialise_Experiance (self):
        # Online training of the network per bellmans equation requires the use of the network to
        # return the quality of the subsequent state, hence this process requires the network to
        # have undertaken an initial fitting, otherwise it will give no result. Additionally the
        # experiance pools must be initialised with some experiance. Initial fitting is conducted
        # using a purely random policy, and continues until the expericance pools have reached their
        # minimum size requirements. Afterwards the network is initialised based upon the immediate
        # reward recieved form each state visited in the initialisation.


        j = 0
        pool = mp.Pool(mp.cpu_count())
        while len(self.Exp) < self.Min_Exp or (len(self.Pos_Exp) < self.Min_Pos_Exp and self.Over_Sample == True) :

            # Map N workers to complete M episodes each, then coalate the N*M episodes and check if
            # the minimum pool sizes have been met.
            Mined_Exp = pool.map(self.Initialisation_Worker, range(self.Init_Async_Settings[1]))

            for i in range(len(Mined_Exp)):
                for k in range(len(Mined_Exp[i])):
                    if Mined_Exp[i][k][1] != None and len(self.Pos_Exp) < self.Min_Pos_Exp and self.Over_Sample == True:
                        self.Pos_Exp.extend(Mined_Exp[i][k][1])
                    elif len(self.Exp) < self.Min_Exp and Mined_Exp[i][k][0] != None:
                        self.Exp.extend(Mined_Exp[i][k][0])

            # Print the number of episodes which have been sampled, and the pool sizes.
            j += self.Init_Async_Settings[0] * self.Init_Async_Settings[1]
            print("Episode " + str(j) + "    Exp Pool : " + str(len(self.Exp)) + "    Pos_Exp Pool : " + str(len(self.Pos_Exp)))


        # next check if the model is new, or has been loaded from a previous runtime
        if self.Model_Loaded == False:
            # For the first step, train based upon immediate reward alone
            Data = np.array(self.Exp + self.Pos_Exp)

            States = list(Data[:,0])
            Actions = list(Data[:,1])
            X = [States[i] + Actions[i] for i in range(len(Data))]
            Y = list(Data[:,2])

            self.Online_Model.fit(X, Y)


    def Initialisation_Worker(self, _):
        # This function is ran by each worker. They will sample M episodes and return the data

        Env = gym.make(self.Env_String)
        Worker_Exp = []

        for i in range(self.Init_Async_Settings[0]):
            State_0 = Env.reset()
            Done = False
            Episode_Exp = []
            while Done == False:
                Action = Env.action_space.sample()
                State_1, Reward, Done, Info = Env.step(Action)

                if Done == False:
                    Episode_Exp.append([list(State_0), list(Action), Reward, list(State_1)])
                    State_0 = copy.deepcopy(State_1)
                else:
                    Episode_Exp.append([list(State_0), list(Action), Reward, [None]])
                    State_0 = copy.deepcopy(State_1)

            # The None trick below allows the parent fucntion to be able to discern whether an episide ended
            # in success or failiure, so positive expericance may be oversampled if required.
            if Reward > self.Positive_Level and len(self.Pos_Exp) < self.Min_Pos_Exp and self.Over_Sample == True:
                Worker_Exp.append([None, Episode_Exp])
            elif len(self.Exp) < self.Min_Exp:
                Worker_Exp.append([Episode_Exp, None])
            else:
                Worker_Exp.append([None, None])

        return Worker_Exp


    def Argmax_Action (self, State):
        # This function checks the quality of a descrtised sample of the continuous
        # action space for a given state and returns the best action in the sample.
        # Since the actionspace is contunious it is impossible to check all actions,
        # and hence the action returned may not be truely optimal.
        # Actor critic seeks to solve this problem to allow for a truely continuous action space.

        if State[0] == None:
            return self.Env.action_space.sample(), 0.0

        Max = self.Env.action_space.high
        Min = self.Env.action_space.low
        dx = (Max - Min) / (self.ArgMax_Steps - 1)

        Actions = np.arange(self.ArgMax_Steps) * dx + Min
        Actions[-1] = Max
        State = list(State)
        State = [State] * self.ArgMax_Steps
        State = np.array(State)
        State1 = np.hstack((State, Actions.reshape(self.ArgMax_Steps, 1)))
        Action_Qualities = self.Online_Model.predict(State1)

        Max_Index  = np.argmax(Action_Qualities)
        Opt_Action = Actions[Max_Index]
        Opt_Reward = Action_Qualities[Max_Index]

        return [Opt_Action], Opt_Reward


    def Train (self, N_Episodes, N_Fully_Random = 0, Epsilon_Override = None):

        # N_Episodes       - Number of episodes to train for
        # N_Fully_Random   - Number of episodes to follow a random policy (Optional)
        # Epsilon_Override - Override the decaying Epsilon with this value throughout training (Optional)

        # This function instigates the training process, adding experiance to the experiance pools
        # and updating the value function network as required.

        # Check if the expereience pool has been initialised
        if len(self.Exp) < self.Min_Exp:
            self.Initialise_Experiance()

        Episode_Rewards   = []
        Episode_Epsilons  = []
        Recent_Exp        = []

        for i in range(N_Episodes):

            # Calculate the episodes Epsilon.
            if Epsilon_Override == None:
                Epsilon = math.exp(-self.Model_Iter_ / 100) if (i >= N_Fully_Random) else 1.0
            else:
                Epsilon = Epsilon_Override

            State_0 = self.Env.reset()
            Done = False

            # If the network is due an update, update it now.
            if i % self.Update_Network == 0:
                if len(Recent_Exp) > 0:         # As the above will be true when i == 0
                    self.Train_Network(Recent_Exp)
                    Recent_Exp = []


            Episode_Experiance = []
            while Done == False:

                # Draw the epsilon greedy action
                if np.random.uniform(0,1) <= Epsilon:
                    Action = self.Env.action_space.sample()
                else:
                    Action, _ = self.Argmax_Action(State_0)

                # Step forwards and update the experiance pools.
                State_1, Reward, Done, Info = self.Env.step(Action)
                if Done == False:
                    Episode_Experiance.append([list(State_0), list(Action), Reward, list(State_1)])
                    Recent_Exp.append([list(State_0), list(Action), Reward, list(State_1)])
                    State_0 = State_1
                else:
                    # If State_1 is terminal None is appended in its place, then the Argmax function
                    # knows this is the terminal state and it must have a future value of zero, otherwise
                    # the value of the termianl state grow continuously.
                    Episode_Experiance.append([list(State_0), list(Action), Reward, [None]])
                    Recent_Exp.append([list(State_0), list(Action), Reward, [None]])
                    State_0 = State_1


            if Reward > self.Positive_Level and self.Over_Sample == True:
                self.Pos_Exp.extend(Episode_Experiance)
            self.Exp.extend(Episode_Experiance)

            Episode_Rewards.append(Reward)
            Episode_Epsilons.append(Epsilon)
            print(str(i), end = " ")

        return Episode_Rewards, Episode_Epsilons


    def Train_Network (self, Recent_Exp):
        # Recent_Exp - The experiance from the most recent episodes, since the lst training session.

        # This function creates a dataset to train the DQN on by sampling data from the varoius pools
        # according to whether Over_Sample and or Train_Recent_Exp are set to True

        if self.Over_Sample == True:
            if self.Train_Recent_Exp == True:
                N = int(min(self.Replay_Size / 2, len(self.Pos_Exp), len(self.Exp)))
                Data = random.sample(self.Exp, N)
                Data.extend(random.sample(self.Pos_Exp, N))
                Data.extend(Recent_Exp)
                Data = np.array(Data)
            else:
                N = int(min(self.Replay_Size / 2, len(self.Pos_Exp), len(self.Exp)))
                Data = random.sample(self.Exp, N)
                Data.extend(random.sample(self.Pos_Exp, N))
                Data = np.array(Data)

        else:
            if self.Train_Recent_Exp == True:
                N = int(min(self.Replay_Size, len(self.Exp)))
                Data = random.sample(self.Exp, N)
                Data.extend(Recent_Exp)
                Data = np.array(Data)
            else:
                N = int(min(self.Replay_Size, len(self.Exp)))
                Data = random.sample(self.Exp, N)
                Data = np.array(Data)

        # Once the dataset 'Data' has been generated, the required information can be extracted,
        # and the value of each state per bellmans equation (using the 'frozen' or old copy of
        # the DQN) can be calculated.

        States        = list(Data[:,0])
        Actions       = list(Data[:,1])
        Future_States = list(Data[:,3])

        X = [States[i] + Actions[i] for i in range(len(Data))]
        Y = list(Data[:,2])

        for i, State in enumerate(Future_States):
            _ , Value = self.Argmax_Action(State)
            Y[i] += Value * self.Gamma

        self.Model_Iter_ += 1
        self.Online_Model.fit(X, Y)




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
                _, Value = self.Argmax_Action(State)
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
                Action, _ = self.Argmax_Action(State)
                V_Function.append([x_low + i * dx, y_low + j * dy, Action[0]])
        V_Function = np.array(V_Function)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(V_Function[:,0], V_Function[:,1], V_Function[:,2], zdir='z', c= 'red')
        plt.show()


    def Decide_Action (self, State):
        Action, _ = self.Argmax_Action(State)
        return Action


    def Save (self, FileName):
        with open(FileName, 'wb') as file:
            pickle.dump([self.Online_Model, self.Model_Iter_], file)


    def Load (self, FileName):
        with open(FileName, 'rb') as file:
            Loaded_Data = pickle.load(file)
            self.Online_Model = Loaded_Data[0]
            self.Model_Iter_  = Loaded_Data[1]
        self.Model_Loaded = True



















# End
