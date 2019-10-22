import numpy             as np
import tqdm
import warnings

def Empty (*args):
    ''' An empty function which accepts any number of arguments '''
    pass

# A Neural Network, with custom loss function for Mu only policy gradient
class Policy_NeuralNet:

    def __init__ (self, Shape, Input_Dim, Output_Dim, Learning_Rate = 0.01, Epoch = 1, Activation = "Relu", Alpha = 0.0001):

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
        self.Batch_Size    = Batch_Size
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

    def Sigmoid(self, X):
        return 1.0 / (1.0 + np.exp(-X))

    def d_Sigmoid(self, X):
        return self.Sigmoid(X) * (1 - self.Sigmoid(X))

    def Relu(self, X):
        return X * (X > 0)

    def d_Relu(self, X):
        return (X > 0)

    def d_Loss(self, Action, Mu, Reward, Sigma):
        return -((Action - Mu) / (Sigma ** 2)) * Reward

    def Forward_Pass(self, X):
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
        self.Forward_Pass(State)
        return self.Zs[-1]

    def Fit(self, State, Action, Reward, Sigma, LR_Mult = 1):
        self.LR_Mult = LR_Mult

        for _ in range(self.Epoch):

            if hasattr(self, 'Learning_Rate_Index'):
                self.Learning_Rate_Index += 1
                if self.Learning_Rate_Index == self.Learning_Rate_Cycle.size:
                    self.Learning_Rate_Index = 0

            if self.Batch_Size == 0:
                self.Forward_Pass(State)
                self.BackProp(State, Action, Reward, Sigma)

            else:
                idx = np.random.choice(State.shape[0], size = (State.shape[0] // self.Batch_Size, self.Batch_Size), replace = False)
                for i in range(idx.shape[0]):
                    self.Forward_Pass(State[idx[i]])
                    self.BackProp(State[idx[i]], Action[idx[i]], Reward[idx[i]], Sigma)



class Policy_Gradient:

    def __init__ (self, Environment, Network_Hypers, Gamma = 0.98, Sigma_Range = [2, 0.2], Sigma_Anneal = 0.75, Retrain_Frequency = 50):

        self.Environment = Environment

        self.Gamma             = Gamma
        self.State_Dim         = Environment.observation_space.shape[0]
        self.Action_Dim        = Environment.action_space.shape[0]
        self.Sigma_Range       = Sigma_Range
        self.Sigma_Anneal      = Sigma_Anneal
        self.Network_Hypers    = Network_Hypers
        self.Retrain_Frequency = Retrain_Frequency

        self.Network = Policy_NeuralNet(self.Network_Hypers["Network Size"], self.State_Dim, self.Action_Dim,
                                        Learning_Rate = self.Network_Hypers["Learning Rate"],
                                        Activation    = self.Network_Hypers["Activation"],
                                        Batch_Size    = self.Network_Hypers["Batch Size"],
                                        Epoch         = self.Network_Hypers["Epoch"],
                                        Alpha         = self.Network_Hypers["Alpha"])


    def Train (self, N_Episodes, Plot = Empty):
        '''
        Train the Policy Grad agent in its specified environment.

        Parameters
        ----------
            N_Episodes : The number of episodes the agent should be trained for. Note parameters like
                         sigma and learning rate decay scale with the number of episodes.

            Plot       : A funtion pointer to the plot generation function (included in the wrapper.)
        '''

        Exp = []
        Plot_Exps = []
        Step_Count = 0

        for i in tqdm.tqdm(range(N_Episodes)):
            Episode_Exp = []
            Sigma = max(self.Sigma_Range[0] - ((self.Sigma_Range[0] - self.Sigma_Range[1]) * (i / (self.Sigma_Anneal * N_Episodes))), self.Sigma_Range[1])
            State_0 = self.Environment.reset()

            Done = False
            while Done == False:
                Mu = self.Network.Predict(State_0.reshape(1, -1)).flatten()
                Leverage = np.random.normal(Mu, Sigma)

                State_1, Reward, Done, Info = self.Environment.step(Leverage)
                Episode_Exp.append({"s0" : State_0, "s1" : State_1, "r" : Reward, "a" : Leverage, 'i' : Info, 'Mu' : Mu, 'd' : Done})
                State_0 = State_1
                Step_Count += 1

            Exp.extend(Episode_Exp)
            Plot_Exps.append(Episode_Exp)

            if Step_Count > 10000:
                Plot(Plot_Exps)
                Plot_Exps = []
                Step_Count = 0

            if i % self.Retrain_Frequency == 0:
                State_0 = np.array([e['s0'] for e in Exp]).reshape((-1, self.State_Dim))
                State_1 = np.array([e['s1'] for e in Exp]).reshape((-1, self.State_Dim))
                Action  = np.array([e['a']  for e in Exp]).reshape((-1, self.Action_Dim))
                Reward  = np.array([e['r']  for e in Exp]).reshape((-1, 1))
                Done    = np.array([e['d']  for e in Exp]).reshape((-1, 1)).astype(int)
                Value   = Reward + self.Gamma * Done * self.Critic.Predict(State_1)

                Advantage = Value - self.Critic.Predict(State_0)

                if self.MiniBatch_Size == 0:
                    self.Actor.Fit(State_0, Action, Advantage, Sigma, LR_Mult = Sigma)
                    self.Critic.Fit(State_0, Value)

                else:
                    idx = np.random.choice(State_0.shape[0], size = (State_0.shape[0] // self.MiniBatch_Size, self.MiniBatch_Size), replace = False)
                    for i in range(idx.shape[0]):
                        self.Actor.Fit(State_0[idx[i]], Action[idx[i]], Advantage[idx[i]], Sigma, LR_Mult = Sigma)
                        self.Critic.Fit(State_0[idx[i]], Reward[idx[i]])

                Exp = []
