
import numpy as np
import tqdm


def Empty (*args):
    ''' An empty function which accepts any number of arguments '''
    pass

# A fully connected Neural Network, with squared loss.
class Neural_Net:

    def __init__ (self, Shape, Input_Dim, Output_Dim, Learning_Rate = 0.01, Batch_Size = 200, Epoch = 1, Activation = "Relu", Alpha = 0.005):

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
        self.Batch_Size    = Batch_Size
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
        return 1.0 / (1.0 + np.exp(-X))

    def d_Sigmoid (self, X):
        return self.Sigmoid(X) * (1 - self.Sigmoid(X))

    def Relu (self, X):
        return X * (X > 0)

    def d_Relu (self, X):
        return (X > 0)

    def d_Loss (self, Y_hat, Y):
        return 2 * (Y_hat - Y)

    def Forward_Pass (self, X):
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
        self.Forward_Pass(X)
        return self.Zs[-1]

    def Fit (self, X, Y):
        for _ in range(self.Epoch):

            if hasattr(self, 'Learning_Rate_Index'):
                self.Learning_Rate_Index += 1
                if self.Learning_Rate_Index == self.Learning_Rate_Cycle.size:
                    self.Learning_Rate_Index = 0

            if self.Batch_Size == 0:
                self.Forward_Pass(X)
                self.BackProp(X, Y)

            else:
                idx = np.random.choice(X.shape[0], size = (X.shape[0] // self.Batch_Size, self.Batch_Size), replace = False)
                for i in range(idx.shape[0]):
                    self.Forward_Pass(X[idx[i]])
                    self.BackProp(X[idx[i]], Y[idx[i]])


class DQN_Agent ():

    def __init__ (self, Environmentironment, Network_Hypers, Action_Disc = 11, Gamma = 0.98, Retrain_Frequency = 50, Retrain_Batch = int(1e4),  Epsilon_Range = [1, 0.1], Epsilon_Anneal = 0.1, Max_Exp = 1e6):

        self.Environment = Environmentironment

        assert self.Environment.action_space.shape[0] == 1, 'DQN may only be used with a one dimensional action space.'

        self.Gamma             = Gamma
        self.Max_Exp           = Max_Exp
        self.State_Dim         = self.Environment.observation_space.shape[0]
        self.Action_Dim        = self.Environment.action_space.shape[0]
        self.Action_Disc       = Action_Disc
        self.Action_Space      = np.linspace(self.Environment.action_space.low, self.Environment.action_space.high, self.Action_Disc).flatten()
        self.Retrain_Batch     = Retrain_Batch
        self.Epsilon_Range     = Epsilon_Range
        self.Epsilon_Anneal    = Epsilon_Anneal
        self.Network_Hypers    = Network_Hypers
        self.Retrain_Frequency = Retrain_Frequency

        self.Network = Neural_Net(self.Network_Hypers["Network Size"], self.State_Dim + self.Action_Dim, 1,
                                  Learning_Rate = self.Network_Hypers["Learning Rate"],
                                  Activation    = self.Network_Hypers["Activation"],
                                  Batch_Size    = self.Network_Hypers["Batch Size"],
                                  Epoch         = self.Network_Hypers["Epoch"],
                                  Alpha         = self.Network_Hypers["Alpha"])

        self.Exp = []


    def Train (self, N_Episodes, Plot = Empty, Diag = Empty):

        epsilons = self.Epsilon_Range[0] * np.ones(N_Episodes) - (1 / self.Epsilon_Anneal) * np.arange(0, N_Episodes) * (self.Epsilon_Range[0] - self.Epsilon_Range[1])
        epsilons = np.maximum(epsilons, self.Epsilon_Range[1])

        Step_Count = 0
        Plot_Exps  = []

        for i in tqdm.tqdm(range(N_Episodes)):

            State_0 = self.Environment.reset()
            Episode_Exp = []

            Done = False
            while Done == False:
                Mu = self.Predict_Action(State_0.reshape(1,-1))
                if np.random.uniform() < epsilons[i]:
                    Action = np.random.choice(self.Action_Space)
                else:
                    Action = Mu

                State_1, Reward, Done, Info = self.Environment.step(np.array([Action]))
                Episode_Exp.append({"s0" : State_0, "s1" : State_1, "r" : Reward, "a" : Action, "i" : Info, "d" : Done, 'Mu' : Mu})
                State_0 = State_1
                Step_Count += 1

                self.Exp.extend(Episode_Exp)
                if len(self.Exp) > self.Max_Exp:
                    self.Exp[0:int(self.Max_Exp * 0.1)] = []
                Plot_Exps.append(Episode_Exp)

                if Step_Count > 5000:
                    Plot(Plot_Exps)
                    Plot_Exps  = []
                    Step_Count = 0

            # Refit the model
            if i % self.Retrain_Frequency == 0 and len(self.Exp) > self.Retrain_Batch:
                Data = np.random.choice(self.Exp, size = self.Retrain_Batch, replace = False)

                X = np.array([d['s0'] for d in Data]).reshape((-1, self.State_Dim))
                X = np.hstack((X, np.array([d['a'] for d in Data]).reshape((-1, self.Action_Dim))))

                S1 = np.array([d['s1'] for d in Data]).reshape((-1, self.State_Dim))
                S1_Val = np.zeros((S1.shape[0], 1))
                for j in range(S1.shape[0]):
                    S1_Val[j,0] = self.Predict_Value(S1[j].reshape(1,-1))

                Y = np.array([d['r'] for d in Data]).reshape(-1,1)
                Y = Y + S1_Val * self.Gamma * np.array([(d['d'] == False) for d in Data]).reshape(-1,1)

                self.Network.Fit(X, Y)


    def Predict_Value (self, State):

        assert State.shape[0] == 1, 'Please send each state individually'

        State = np.repeat(State, self.Action_Disc, axis = 0)
        State = np.hstack((State, self.Action_Space.reshape(-1,1)))

        Quality = self.Network.Predict(State).flatten()

        return np.max(Quality)


    def Predict_Action (self, State):
        ''' A DQN of this specification predicts an action by argmaxing across its Action_Space '''

        assert State.shape[0] == 1, 'Please send each state individually'

        State = np.repeat(State, self.Action_Disc, axis = 0)
        State = np.hstack((State, self.Action_Space.reshape(-1,1)))

        Quality = self.Network.Predict(State).flatten()
        return self.Action_Space[np.argmax(Quality)]
