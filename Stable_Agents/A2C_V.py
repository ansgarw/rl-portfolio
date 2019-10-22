import numpy as np
import tqdm
import warnings

def Empty (*args):
    ''' An empty function which accepts any number of arguments '''
    pass


# A Neural Network, with squared loss.
class Critic_NeuralNet:

    def __init__ (self, Shape, Input_Dim, Output_Dim, Learning_Rate = 0.01, Epoch = 1, Activation = "Relu", Alpha = 0.005):

        '''
        The Critic neural network uses a standard SSE loss function

        Parameters
        ----------
            Shape | list
                The number of nodes to put in each hidden layer.

            Input_Dim | int
                The number of variables in a single observation of the input.

            Output_Dim | int
                The number of variables in a single obervation of the output.

            Learning_Rate | float, list
                The network learning rate if passed as a float. If a list is passed it must contain 2 floats followed by an int, in the following configuration [Min_Learning_Rate, Max_Learning_Rate, Cycle_Length]. The network will then utilise cyclical learning rates, cycling between the two bounds.

            Epoch | int
                The number of times the data should be passed through the network per call to .Fit. Using a low learning rate with higher Epoch may smooth performance.

            Activation | one of ['Relu', 'Sigmoid']
                The activation function.

            Alpha | float
                L2 regularization coefficient. Set to zero to disable.

        '''

        if isinstance(Learning_Rate, list):
            assert len(Learning_Rate) == 3, 'Learning_Rate must be a float or a list of lenght 3'
            self.useCyclical_LR = True
            self.Learning_Rate = Learning_Rate[:2]
            self.Learning_Rate_Cycle = np.append(np.linspace(Learning_Rate[0], Learning_Rate[1], Learning_Rate[2]/2), np.linspace(Learning_Rate[1], Learning_Rate[0], Learning_Rate[2]/2))
            self.Learning_Rate_Index = 0
        else:
            self.Learning_Rate = Learning_Rate

        self.Weights = list()
        self.Biases  = list()
        self.Output_Dim    = Output_Dim
        self.Epoch         = Epoch
        self.Alpha         = Alpha

        if Activation == "Relu":
            self.Act   = self.Relu
            self.d_Act = self.d_Relu
        elif Activation == "Sigmoid":
            self.Act   = self.Sigmoid
            self.d_Act = self.d_Sigmoid

        self.Shape = [Input_Dim] + Shape + [Output_Dim]
        for i in range(1, len(self.Shape)):
            self.Weights.append(np.random.normal(0, 1, (self.Shape[i-1], self.Shape[i])) * ((2 / (self.Shape[i-1] + self.Shape[i])) ** 0.5))
            self.Biases.append(np.random.normal(0, 1, (self.Shape[i], 1)))

        self.As      = [None] * (len(self.Weights) + 1)   #Pre Sigmoid
        self.Zs      = [None] * (len(self.Weights) + 1)   #Post Sigmoid

    def Sigmoid (self, X):
        ''' The sigmoid activation function. Accepts float or np.array '''
        return 1.0 / (1.0 + np.exp(-X))

    def d_Sigmoid (self, X):
        ''' The derivative of the sigmoid activation function. Accepts float or np.array '''
        return self.Sigmoid(X) * (1 - self.Sigmoid(X))

    def Relu (self, X):
        ''' The Relu activation function. Accepts float or np.array '''
        return X * (X > 0)

    def d_Relu (self, X):
        ''' The derivative of the Relu activation function. Accepts float or np.array '''
        return (X > 0)

    def d_Loss (self, Y_hat, Y):
        ''' The derivative of the SSE loss function. '''
        return 2 * (Y_hat - Y)

    def Forward_Pass (self, X):
        ''' Conduct a forward pass through the network, and store the result in self.Zs (Network activations.) '''
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

    def BackProp (self, X, Y):
        ''' Backpropergation, using the activations stored in self.Zs '''
        Grads = []
        for i in range(len(self.Weights))[::-1]:
            A      = self.As[i+1]
            Z      = self.Zs[i+1]
            Z_Prev = self.Zs[i]
            W      = self.Weights[i]

            dA = (self.d_Loss(Z, Y)).T if i+1 == len(self.Weights) else (self.d_Act(A).T * dZ)

            # get parameter gradients
            dW = (np.matmul(dA, Z_Prev) / X.shape[0]) + ((self.Alpha * W.T) / X.shape[0])
            dB = np.sum(dA, axis=1).reshape(-1,1) / X.shape[0]
            Grads.append({'Weight' : dW, 'Bias' : dB})

            if i > 0:
                dZ = np.dot(W, dA)

        Grads = Grads[::-1]
        for i in range(len(Grads)):
            if hasattr(self, 'Learning_Rate_Cycle'):
                self.Weights[i] -= self.Learning_Rate_Cycle[self.Learning_Rate_Index] * Grads[i]["Weight"].T
                self.Biases[i]  -= self.Learning_Rate_Cycle[self.Learning_Rate_Index] * Grads[i]["Bias"]
            else:
                self.Weights[i] -= self.Learning_Rate * Grads[i]["Weight"].T
                self.Biases[i]  -= self.Learning_Rate * Grads[i]["Bias"]

    def Predict (self, X):
        '''
        User facing function to extract a prediction from the network.

        Parameters
        ----------
        X | np.array (2D)
            A 2D array with obervations in the rows (i.e. each row contains a new observation)

        Returns
        -------
            np.array (2D)
                An array with the network predictions for each observation in X.
        '''

        self.Forward_Pass(X)
        return self.Zs[-1]

    def Fit (self, X, Y):
        '''
        User facing function to fit the network.

        Parameters
        ----------
        X | np.array (2D)
            A 2D array with obervations in the rows (i.e. each row contains a new observation)

        Y | np.array (2D)
            A 2D array with outcomes in the rows

        '''

        for _ in range(self.Epoch):

            if hasattr(self, 'Learning_Rate_Index'):
                self.Learning_Rate_Index += 1
                if self.Learning_Rate_Index == self.Learning_Rate_Cycle.size:
                    self.Learning_Rate_Index = 0

            self.Forward_Pass(X)
            self.BackProp(X, Y)


# A Neural Network, with custom loss function for Mu only policy gradient
class Policy_NeuralNet:

    def __init__ (self, Shape, Input_Dim, Output_Dim, Learning_Rate = 0.01, Epoch = 1, Activation = "Relu", Alpha = 0.0001):

        '''
        The Critic neural network uses a standard SSE loss function

        Parameters
        ----------
            Shape | list
                The number of nodes to put in each hidden layer.

            Input_Dim | int
                The number of variables in a single observation of the input.

            Output_Dim | int
                The number of variables in a single obervation of the output.

            Learning_Rate | float, list
                The network learning rate if passed as a float. If a list is passed it must contain 2 floats followed by an int, in the following configuration [Min_Learning_Rate, Max_Learning_Rate, Cycle_Length]. The network will then utilise cyclical learning rates, cycling between the two bounds.

            Epoch | int
                The number of times the data should be passed through the network per call to .Fit. Using a low learning rate with higher Epoch may smooth performance.

            Activation | one of ['Relu', 'Sigmoid']
                The activation function.

            Alpha | float
                L2 regularization coefficient. Set to zero to disable.

        '''

        if isinstance(Learning_Rate, list):
            assert len(Learning_Rate) == 3, 'Learning_Rate must be a float or a list of lenght 3'
            self.useCyclical_LR = True
            self.Learning_Rate = Learning_Rate[:2]
            self.Learning_Rate_Cycle = np.append(np.linspace(Learning_Rate[0], Learning_Rate[1], Learning_Rate[2]/2), np.linspace(Learning_Rate[1], Learning_Rate[0], Learning_Rate[2]/2))
            self.Learning_Rate_Index = 0
        else:
            self.Learning_Rate = Learning_Rate

        self.Weights = list()
        self.Biases  = list()
        self.Output_Dim    = Output_Dim
        self.Epoch         = Epoch
        self.Alpha         = Alpha
        self.LR_Mult       = 1

        if Activation == "Relu":
            self.Act   = self.Relu
            self.d_Act = self.d_Relu
        elif Activation == "Sigmoid":
            self.Act   = self.Sigmoid
            self.d_Act = self.d_Sigmoid

        self.Shape = [Input_Dim] + Shape + [Output_Dim]
        for i in range(1, len(self.Shape)):
            self.Weights.append(np.random.normal(0, 1, (self.Shape[i-1], self.Shape[i])) * ((2 / (self.Shape[i-1] + self.Shape[i])) ** 0.5))
            self.Biases.append(np.random.normal(0, 1, (self.Shape[i], 1)))

        self.As      = [None] * (len(self.Weights) + 1)   #Pre Sigmoid
        self.Zs      = [None] * (len(self.Weights) + 1)   #Post Sigmoid

    def Sigmoid (self, X):
        ''' The sigmoid activation function. Accepts float or np.array '''
        return 1.0 / (1.0 + np.exp(-X))

    def d_Sigmoid (self, X):
        ''' The derivative of the sigmoid activation function. Accepts float or np.array '''
        return self.Sigmoid(X) * (1 - self.Sigmoid(X))

    def Relu (self, X):
        ''' The Relu activation function. Accepts float or np.array '''
        return X * (X > 0)

    def d_Relu (self, X):
        ''' The derivative of the Relu activation function. Accepts float or np.array '''
        return (X > 0)

    def d_Loss(self, Action, Mu, Reward, Sigma):
        ''' The derivative of the log likelihood of picking Action from a normal distribution of N(Mu, Sigma), multiplied by Advantage '''
        return -((Action - Mu) / (Sigma ** 2)) * Reward

    def Forward_Pass(self, X):
        ''' Conduct a forward pass through the network, and store the result in self.Zs (Network activations.) '''
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

    def BackProp(self, State, Action, Reward, Sigma):
        ''' Backpropergation, using the activations stored in self.Zs '''
        Grads = []
        for i in range(len(self.Weights))[::-1]:
            A      = self.As[i+1]
            Z      = self.Zs[i+1]
            Z_Prev = self.Zs[i]
            W      = self.Weights[i]

            dA = (self.d_Loss(Action, Z, Reward, Sigma)).T if i+1 == len(self.Weights) else (self.d_Act(A).T * dZ)

            # get parameter gradients
            dW = (np.matmul(dA, Z_Prev) / State.shape[0]) + ((self.Alpha * W.T) / State.shape[0])
            dB = np.sum(dA, axis=1).reshape(-1,1) / State.shape[0]
            Grads.append({'Weight' : dW, 'Bias' : dB})

            if i > 0:
                dZ = np.dot(W, dA)

        Grads = Grads[::-1]
        for i in range(len(Grads)):
            if hasattr(self, 'Learning_Rate_Cycle'):
                self.Weights[i] -= self.Learning_Rate_Cycle[self.Learning_Rate_Index] * self.LR_Mult * Grads[i]["Weight"].T
                self.Biases[i]  -= self.Learning_Rate_Cycle[self.Learning_Rate_Index] * self.LR_Mult * Grads[i]["Bias"]
            else:
                self.Weights[i] -= self.Learning_Rate * self.LR_Mult * Grads[i]["Weight"].T
                self.Biases[i]  -= self.Learning_Rate * self.LR_Mult * Grads[i]["Bias"]

    def Predict(self, State):
        '''
        User facing function to extract a prediction from the network.

        Parameters
        ----------
        State | np.array (2D)
            A 2D array with obervations in the rows (i.e. each row contains a new observation)

        Returns
        -------
            np.array (2D)
                An array with the network predictions for each observation in State.
        '''

        self.Forward_Pass(State)
        return self.Zs[-1]

    def Fit(self, State, Action, Reward, Sigma, LR_Mult = 1):
        '''
        User facing function to fit the network.

        Parameters
        ----------
        State | np.array (2D)
            A 2D array with State_0 in the rows (i.e. each row contains a new observation)

        Action | np.array (2D)
            A 2D array with Actions in the rows (i.e. each row contains a new observation)

        Reward | np.array (2D)
            A 2D array with Rewards in the rows (i.e. each row contains a new observation)

        Sigma | float
            The Sigma of the policy. This is not decided by the network in this version, but instead is slowly decayed across the learning episodes.

        LR_Mult | float
            In order to counter the loss function exploding with decreasing sigma the Learning_Rate may be decayed using this parameter.

        '''

        self.LR_Mult = LR_Mult

        if hasattr(self, 'Learning_Rate_Index'):
            self.Learning_Rate_Index += 1
            if self.Learning_Rate_Index == self.Learning_Rate_Cycle.size:
                self.Learning_Rate_Index = 0

        for _ in range(self.Epoch):
            self.Forward_Pass(State)
            self.BackProp(State, Action, Reward, Sigma)



class Actor_Critic:

    def __init__ (self, Environment, Actor_Hypers, Critic_Hypers, Gamma, Sigma_Range, Sigma_Anneal, Retrain_Frequency, MiniBatch_Size = 0):

        '''
        Parameters
        ----------
            Environment | OpenAI Gym enviornment
                The environment to train the AC within, ususally one of the Portfolio_Gym environments

            Gamma | float
                The discount rate for reward recieved by the agent.

            Sigma_Range | list
                A list of two floats, the first gives the starting sigma, and the last giving the terminal sigma. Sigma here referes to the sigma of the policy.

            Sigma_Anneal | float
                The fraction of training episodes which must pass before sigma decays to its terminal value.

            Retrain_Frequency | int
                The number of episodes between refits of the Actor and Critic

            MiniBatch_Size | int
                The size of the batch of data the networks are fitted on. Set to zero to disable minibatching. (it is reccomended to set this parameter to zero and simply refit more frequently.)


            Actor_Hypers | dict
                'Network Size'  | int         | The size of the Actor NN
                'Learning Rate' | float, list | The learning rate of the Actor. list for cyclical learning rate, see network annotation.
                'Activation'    | string      | The activation of the Actor. Acceptable inputs include ['Relu', 'Sigmoid']
                'Epoch'         | int         | The Actor Epoch
                'Alpha'         | float       | L2 regularization coefficient for the Actor.

            Critic_Hypers | dict
                'Network Size'  | int         | The size of the Critic NN
                'Learning Rate' | float, list | The learning rate of the Critic. list for cyclical learning rate, see network annotation.
                'Activation'    | string      | The activation of the Critic. Acceptable inputs include ['Relu', 'Sigmoid']
                'Epoch'         | int         | The Critic Epoch
                'Alpha'         | float       | L2 regularization coefficient for the Critic.

        '''

        self.Retrain_Frequency = Retrain_Frequency
        self.Sigma_Range       = Sigma_Range
        self.Sigma_Anneal      = Sigma_Anneal
        self.Gamma             = Gamma
        self.State_Dim         = Environment.observation_space.shape[0]
        self.Action_Dim        = Environment.action_space.shape[0]
        self.MiniBatch_Size    = MiniBatch_Size

        self.Actor_Hypers  = Actor_Hypers
        self.Critic_Hypers = Critic_Hypers

        self.Actor  = Policy_NeuralNet(self.Actor_Hypers["Network Size"], self.State_Dim, self.Action_Dim,
                                       Learning_Rate = self.Actor_Hypers["Learning Rate"],
                                       Activation    = self.Actor_Hypers["Activation"],
                                       Epoch         = self.Actor_Hypers["Epoch"],
                                       Alpha         = self.Actor_Hypers["Alpha"])

        self.Critic = Critic_NeuralNet(self.Critic_Hypers["Network Size"], self.State_Dim, 1,
                                       Learning_Rate = self.Critic_Hypers["Learning Rate"],
                                       Activation    = self.Critic_Hypers["Activation"],
                                       Epoch         = self.Critic_Hypers["Epoch"],
                                       Alpha         = self.Critic_Hypers["Alpha"])

        self.Environment = Environment


    def Train (self, N_Episodes, Plot = Empty, Diag = Empty):

        '''
        Train the AC agent in its specified environment.

        Parameters
        ----------
            N_Episodes | int
            The number of episodes the agent should be trained for. Note parameters like sigma and learning rate decay scale with the number of episodes.

            Plot | func
                A function pointer used by the Wrapper to plot the performance of the agent as it learns. This function is called every 10k steps.

            Diag | func
                A function pointer used by the wrapper to plot diagnostics (for example the sensitivity of Actor/Critic to state parameters). This function is called only 5 times throughout training.

        '''

        Exp = []
        Episode_Exps = []
        Step_Count = 0

        for i in tqdm.tqdm(range(N_Episodes)):
            Done = False
            Episode_Exp = []
            State_0 = self.Environment.reset()

            Sigma = max(self.Sigma_Range[0] - ((self.Sigma_Range[0] - self.Sigma_Range[1]) * (i / (self.Sigma_Anneal * N_Episodes))), self.Sigma_Range[1])

            while Done == False:
                Mu = np.clip(self.Actor.Predict(State_0.reshape(1, self.State_Dim)), -10, 10)
                Leverage = np.random.normal(Mu, Sigma)

                State_1, Reward, Done, Info = self.Environment.step(Leverage[0])
                Episode_Exp.append({"s0" : State_0, "s1" : State_1, "r" : Reward, "a" : Leverage, 'i' : Info, 'Mu' : Mu.flatten(), 'd' : Done})
                State_0 = State_1
                Step_Count += 1


            for j in range(len(Episode_Exp) - 1)[::-1]:
                Episode_Exp[j]["r"] += Episode_Exp[j+1]["r"] * self.Gamma

            Exp.extend(Episode_Exp)
            Episode_Exps.append(Episode_Exp)

            if Step_Count > 10000:
                Plot(Episode_Exps)
                Episode_Exps = []
                Step_Count = 0

            if i % int(N_Episodes / 5) == 0:
                Diag()

            if i % self.Retrain_Frequency == 0:
                State_0   = np.array([e['s0'] for e in Exp]).reshape((-1, self.State_Dim))
                Action    = np.array([e['a']  for e in Exp]).reshape((-1, self.Action_Dim))
                Reward    = np.array([e['r']  for e in Exp]).reshape((-1, 1))
                Advantage = Reward - self.Critic.Predict(State_0)

                if self.MiniBatch_Size == 0:
                    self.Actor.Fit(State_0, Action, Advantage, Sigma, LR_Mult = Sigma)
                    self.Critic.Fit(State_0, Reward)

                else:
                    idx = np.random.choice(State_0.shape[0], size = (State_0.shape[0] // self.MiniBatch_Size, self.MiniBatch_Size), replace = False)
                    for i in range(idx.shape[0]):
                        self.Actor.Fit(State_0[idx[i]], Action[idx[i]], Advantage[idx[i]], Sigma, LR_Mult = Sigma)
                        self.Critic.Fit(State_0[idx[i]], Reward[idx[i]])

                Exp = []


    def Predict_Action (self, X):
        '''
        Parameters
        ----------
            X | np.array
                A np array of states or a single state for which the action will be predicted.

        Returns
        -------
            np.array (2D)
                An array of actions (each action on its own row.)
        '''

        return self.Actor.Predict(X.reshape(-1, self.State_Dim))


    def Predict_Value (self, X):
        '''
        Parameters
        ----------
            X | np.array
                A np array of states or a single state for which the value will be predicted.

        Returns
        -------
            np.array (2D)
                An column vector of values.
        '''

        return self.Critic.Predict(X.reshape(-1,self.State_Dim))
