import numpy             as np
import math
import copy
import tqdm


def Empty (*args):
    ''' An empty function which accepts any number of arguments '''
    pass

# A multiheaded Nerual Network with added fucntionality which allows it to train
# towards a single head at a time.
class NeuralNet:
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



# The DQN Agent itself.
class DQN:

    def __init__ (self, Environment, Action_Dim, Network_Params, Gamma = 0.99, Batch_Size = 512, Epsilon_Range = [1, 0.1], Epsilon_Anneal = 0.1, Retrain_Frequency = 100,
                        Max_Exp = 30000):
        '''
        Parameters
        ----------
            Environment       : The gym environment that the agent should train within.
            Action_Dim        : The size of the action_space.
            Network_Params    : A dictionary or parameters used to Initialise the neural network used to approximate the quality function
                                which includes:
                                   Network Size  : A list of N ints where each entry represents the desired number of nodes in the Nth hidden layer
                                   Learning Rate : The learning rate used to update the weights and biases of the network
                                   Activation    : The activation function used. Acceptable inputs include ("Sigmoid", "Relu")
                                   Epoch         : The number of back propergation passes to be ran each time Fit is called.
                                   Alpha         : An L2 regularization term.
            Gamma             : The rate used to discount future reward when appraising the value of a state.
            Batch_Size        : The number of (steps) of experiance to be sent each time .Fit is called on the internal network
            Epsilon_Range     : The range of epsilon (for an epsilon greedy policy) across a training sequence
            Epsilon_Anneal    : The fraction of the training sequence which should have passed for epsilon to fall from its starting
                                value to its terminal value.
            Retrain_Frequency : The frequency at which the internal network is refitted (Measured in episode.)

        '''

        self.Environment = Environment

        self.Epsilon_Range     = Epsilon_Range
        self.Epsilon_Anneal    = Epsilon_Anneal
        self.Retrain_Frequency = Retrain_Frequency
        self.State_Dim         = Environment.observation_space.shape[0]
        self.Action_Dim        = Action_Dim
        self.Action_Space      = np.linspace(Environment.action_space.low, Environment.action_space.high, self.Action_Dim)
        self.Gamma             = Gamma
        self.Batch_Size        = Batch_Size
        self.Network_Params    = Network_Params
        self.Max_Exp           = Max_Exp



        self.Q_Network = NeuralNet(self.Network_Params["Network Size"], self.State_Dim, self.Action_Dim,
                                   Learning_Rate = self.Network_Params["Learning Rate"],
                                   Activation    = self.Network_Params["Activation"],
                                   Epoch         = self.Network_Params["Epoch"],
                                   Alpha         = self.Network_Params["Alpha"],
                                   Batch_Size    = self.Network_Params["Batch Size"])

        self.Exp = []


    def Train (self, N_Episodes, Plot = Empty):

        '''
        Trains the agent

        Parameters
        ----------
            N_Episodes : The number of episodes to train the DQN Agent across.

            Plot       : A list of keywords indicating the plottable metrics which should be returned.
                         Accepted inputs include:
                            1. Merton_Benchmark : Plots the delta between the utility of the Agent vs the Merton Portfolio
                                                  for each episode. To be used only with the historical environment.
                            2. Greedy_Merton    : Plot the average utility of both the Merton Portfolio and the DQN across 100
                                                  episodes (from the training dataset) acting greedily
                            3. Ave_Perf         : Calculates the average terminal reward across 100 episodes after each fitting
                                                  of the neural network.

        Notes
        -----
            Since the DQN learns off policy there should be no negative effects of calling Train() multiple times in series on
            the same agent. However perfered behaviour is to call this function only once as it ensures the agent slowly acts more
            optimally across the training sequence. (Epsilon will jump back to its inital value in subsequent calls to this function).
        '''


        epsilons = self.Epsilon_Range[0] * np.ones(N_Episodes) - (1 / self.Epsilon_Anneal) * np.arange(0, N_Episodes) * (self.Epsilon_Range[0] - self.Epsilon_Range[1])
        epsilons = np.maximum(epsilons, self.Epsilon_Range[1])

        Step_Count = 0
        Episode_Exps = []


        for i in tqdm.tqdm(range(N_Episodes)):

            State_0 = self.Environment.reset()
            Done = False
            Episode_Exp = []

            while Done == False:
                if np.random.uniform() > epsilons[i]:
                    Action_idx = np.random.choice(list(range(self.Action_Dim)))
                else:
                    Action_idx = np.argmax(self.Q_Network.Predict(State_0.reshape(1, self.State_Dim)))

                State_1, Reward, Done, Info = self.Environment.step(self.Action_Space[Action_idx])
                Episode_Exp.append({"s0" : State_0, "s1" : State_1, "r" : Reward, "a" : Action_idx, "i" : Info, "done" : Done, 'Mu' : self.Action_Space[Action_idx]})
                State_0 = State_1
                Step_Count += 1

            self.Exp.extend(Episode_Exp)
            if len(self.Exp) > self.Max_Exp:
                self.Exp[0:len(Episode_Exp)] = []
            Episode_Exps.append(Episode_Exp)

            if Step_Count > 10000:
                Plot(Episode_Exps)
                Episode_Exps = []
                Step_Count = 0

            # Refit the model
            if i % self.Retrain_Frequency == 0 and len(self.Exp) > self.Batch_Size:
                Data = np.random.choice(self.Exp, size = self.Batch_Size, replace = False)
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


    def Predict_Q (self, X):
        ''' A simple pedict fucntion to extract predictions from the agent without having to call methods on the internal network '''
        return self.Q_Network.Predict(X)


    def Predict_Action(self, X):
        ''' Returns the optimal action per the Q Network '''
        return self.Action_Space[np.argmax(self.Q_Network.Predict(X), axis = 1)]
