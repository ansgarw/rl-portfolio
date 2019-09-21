import numpy as np

# A Neural Network, with squared loss.
class Critic_NeuralNet:

    def __init__ (self, Shape, Input_Dim, Output_Dim, Learning_Rate = 0.01, Epoch = 1, Activation = "Relu", Alpha = 0.005):
        self.Weights = list()
        self.Biases  = list()
        self.Learning_Rate = Learning_Rate
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
            self.Weights.append(np.random.normal(0, 1, (self.Shape[i-1], self.Shape[i])) / (self.Shape[i-1] + self.Shape[i]))
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

    def d_Loss(self, Y_hat, Y):
        return 2 * (Y_hat - Y)


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


    def BackProp(self, X, Y):
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
            self.Weights[i] -= self.Learning_Rate * Grads[i]["Weight"].T
            self.Biases[i]  -= self.Learning_Rate * Grads[i]["Bias"]


    def Predict(self, X):
        self.Forward_Pass(X)
        return self.Zs[-1]


    def Fit(self, X, Y):
        for _ in range(self.Epoch):
            self.Forward_Pass(X)
            self.BackProp(X, Y)

# A Neural Network, with custom loss function for Mu only policy gradient
class Policy_NeuralNet:
    def __init__ (self, Shape, Input_Dim, Output_Dim, Learning_Rate = 0.01,
                  Epoch = 1, Activation = "Relu", Alpha = 0.0001):
        self.Weights = list()
        self.Biases  = list()
        self.Learning_Rate = Learning_Rate
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
            self.Weights.append(np.random.normal(0, 1, (self.Shape[i-1], self.Shape[i])))
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
            self.Weights[i] -= self.Learning_Rate * Grads[i]["Weight"].T
            self.Biases[i]  -= self.Learning_Rate * Grads[i]["Bias"]


    def Predict(self, State):
        self.Forward_Pass(State)
        return self.Zs[-1]


    def Fit(self, State, Action, Reward, Sigma):
        for _ in range(self.Epoch):
            self.Forward_Pass(State)
            self.BackProp(State, Action, Reward, Sigma)


class Actor_Critic:

    def __init__ (self, Environment, Actor_Params, Critic_Params, Gamma, Sigma_Range, Sigma_Anneal, Retrain_Frequency):

        self.Retrain_Frequency = Retrain_Frequency
        self.Sigma_Range       = Sigma_Range
        self.Sigma_Anneal      = Sigma_Anneal
        self.Gamma             = Gamma
        self.State_Dim         = Environment.observation_space.shape[0]
        self.Action_Dim        = Environment.action_space.shape[0]
        self.Plot_Frequency    = 5

        self.Actor_Hypers  = Actor_Params
        self.Critic_Hypers = Critic_Params

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

    def Train (self, N_Episodes):

        Exp = []
        Mus = []

        if (self.Plot_Frequency > 0):
            Record_Eps  = np.linspace(0, N_Episodes, self.Plot_Frequency).astype(int) - 1
            Record_Eps[0] = 0
            Plot_Data = []


        for i in range(N_Episodes):
            Done = False
            Episode_Exp = []
            State_0 = self.Environment.reset()

            Sigma = max(self.Sigma_Range[0] - ((self.Sigma_Range[0] - self.Sigma_Range[1]) * (i / (self.Sigma_Anneal * N_Episodes))), self.Sigma_Range[1])
            # Sigma = (Sigma * ((self.Sigma_Range[1] / self.Sigma_Range[0]) ** (1/(self.Sigma_Anneal * 1e5))))
            Actor_LR = self.Actor_Hypers["Learning Rate"] * (Sigma)

            while Done == False:
                Mu = self.Actor.Predict(State_0.reshape(1, self.State_Dim))
                Mus.append(list(Mu.reshape(-1)))
                Leverage = np.random.normal(Mu, Sigma)

                State_1, Reward, Done, Info = self.Environment.step(Leverage[0])
                Episode_Exp.append({"s0" : State_0, "s1" : State_1, "r" : Reward, "a" : Leverage})
                State_0 = State_1


            for j in range(len(Episode_Exp) - 1)[::-1]:
                Episode_Exp[j]["r"] += Episode_Exp[j+1]["r"] * self.Gamma
            Exp.extend(Episode_Exp)


            if i % self.Retrain_Frequency == 0:
                State     = np.array([e['s0'] for e in Exp]).reshape((-1, self.State_Dim))
                Action    = np.array([e['a']  for e in Exp]).reshape((-1, self.Action_Dim))
                Reward    = np.array([e['r']  for e in Exp]).reshape((-1, 1))
                Advantage = Reward - self.Critic.Predict(State)

                self.Actor.Learning_Rate = Actor_LR
                self.Actor.Fit(State, Action, Advantage, Sigma)
                self.Critic.Fit(State, Reward)

                Exp = []

            if self.Plot_Frequency > 0:
                if np.any(i == Record_Eps):
                    Test_State = np.hstack((np.zeros((20,1)), np.linspace(0,1,20).reshape(-1,1)))
                    if self.Action_Dim == 1:
                        Plot_Data.append({"Policy" : self.Actor.Predict(Test_State).reshape(-1),
                                          "Value"  : self.Critic.Predict(Test_State).reshape(-1),
                                          "Title"  : str(i + 1) + " Eps"})

                    else:
                        Plot_Data.append({"Policy" : self.Actor.Predict(Test_State).reshape(-1, self.Action_Dim),
                                          "Value"  : self.Critic.Predict(Test_State).reshape(-1),
                                          "Title"  : str(i + 1) + " Eps"})

        if self.Plot_Frequency > 0:
            return [Mus, Plot_Data]
        else:
            return Mus


    def Validate (self, N_Episodes):
        '''
        A validation function used to appraise the performance of the agent across N episodes.

        Parameters
        ----------
            N_Episodes : The number of episodes to validate across.

        Returns
        -------
            A list of terminal rewards.
        '''

        Terminal_Rewards = []
        self.Environment.isTraining = False

        for i in range(N_Episodes):
            State = self.Environment.reset()
            Done = False

            while Done == False:
                Action = self.Actor.Predict(State.reshape(1, self.State_Dim))
                State, Reward, Done, Info = self.Environment.step(Action.reshape(-1))
                if Done:
                    Terminal_Rewards.append(Reward)

        self.Environment.isTraining = True
        return Terminal_Rewards
