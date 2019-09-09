import numpy as np

# A multiheaded Nerual Network with added fucntionality which allows it to train
# towards a single head at a time.
class NeuralNet:
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


    def BackProp(self, X_, Y_, Z_):

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
        self.Forward_Pass(X)
        return self.Zs[-1]


    def Fit(self, X, Y, Z):
        for _ in range(self.Epoch):
            self.Forward_Pass(X)
            self.BackProp(X, Y, Z)


# The DQN Agent itself.
class DQN:
    def __init__ (self, Environment, Action_Dim, Network_Params, Gamma = 0.99, Batch_Size = 512, Epsilon_Range = [1, 0.1], Epsilon_Anneal = 0.1, Retrain_Frequency = 100):

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


        self.Q_Network = NeuralNet(self.Network_Params["Network Size"], self.State_Dim, self.Action_Dim,
                                   Learning_Rate = self.Network_Params["Learning Rate"],
                                   Activation    = self.Network_Params["Activation"],
                                   Epoch         = self.Network_Params["Epoch"],
                                   Alpha         = self.Network_Params["Alpha"])

        self.Exp = []

    def Train (self, N_Episodes):

        epsilons = self.Epsilon_Range[0] * np.ones(N_Episodes) - (1 / self.Epsilon_Anneal) * np.arange(0, N_Episodes) * (self.Epsilon_Range[0] - self.Epsilon_Range[1])
        epsilons = np.maximum(epsilons, self.Epsilon_Range[1])

        for i in range(N_Episodes):

            State_0 = self.Environment.reset()
            Done = False

            while Done == False:
                if np.random.uniform() > epsilons[i]:
                    Action_idx = np.random.choice(list(range(self.Action_Dim)))
                else:
                    Action_idx = np.argmax(self.Q_Network.Predict(State_0.reshape(1, self.State_Dim)))

                State_1, Reward, Done, Info = self.Environment.step(self.Action_Space[Action_idx])
                self.Exp.append({"s0" : State_0, "s1" : State_1, "r" : Reward, "a" : Action_idx, "done" : Done})
                State_0 = State_1


            if (i + 1) % self.Retrain_Frequency == 0 and len(self.Exp) > self.Batch_Size:
                # Refit the model
                Data = np.random.choice(self.Exp, size = self.Batch_Size, replace = False)
                X  = np.array([d['s0'] for d in Data]).reshape((-1, self.State_Dim))
                Y  = np.array([d['r']  for d in Data])
                Z  = np.array([d['a']  for d in Data]).reshape((-1, 1))
                S1 = np.array([d['s1'] for d in Data]).reshape((-1, self.State_Dim))

                S1_Val = self.Q_Network.Predict(S1)
                S1_Val = np.amax(S1_Val, axis = 1) * np.array([(d['done'] == False) for d in Data])
                Y = (Y + S1_Val * self.Gamma).reshape(-1, 1)

                self.Q_Network.Fit(X, Y, Z)

    def Predict (self, X):
        return self.Q_Network.Predict(X)
