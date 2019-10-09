import numpy             as np
import matplotlib.pyplot as plt
import tqdm


def Empty (*args):
    ''' An empty function which accepts any number of arguments '''
    pass

# Version IV
#   1. Updated the neural networks to use SGD

# A Neural Network, with squared loss.
class Critic_NeuralNet:

    def __init__ (self, Shape, Input_Dim, Output_Dim, Learning_Rate = 0.01, Epoch = 1, Activation = "Relu", Alpha = 0.005, Batch_Size = 200):
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
        if self.Batch_Size > 0:
            for _ in range(self.Epoch):
                idx = np.random.choice(X.shape[0], size = (X.shape[0] // self.Batch_Size, self.Batch_Size), replace = False)
                for i in range(idx.shape[0]):
                    self.Forward_Pass(X[idx[i]])
                    self.BackProp(X[idx[i]], Y[idx[i]])
        else:
            for _ in range(self.Epoch):
                self.Forward_Pass(X)
                self.BackProp(X, Y)

# A Neural Network, with custom loss function for Mu only policy gradient
class Policy_NeuralNet:
    def __init__ (self, Shape, Input_Dim, Output_Dim, Learning_Rate = 0.01, Epoch = 1, Activation = "Relu", Alpha = 0.0001, Batch_Size = 200):
        '''
        Set batch size to zero to disable.
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

        if self.Batch_Size > 0:
            for _ in range(self.Epoch):
                idx = np.random.choice(State.shape[0], size = (State.shape[0] // self.Batch_Size, self.Batch_Size), replace = False)
                for i in range(idx.shape[0]):
                    self.Forward_Pass(State[idx[i]])
                    self.BackProp(State[idx[i]], Action[idx[i]], Reward[idx[i]], Sigma)
        else:
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

        self.Actor_Hypers  = Actor_Params
        self.Critic_Hypers = Critic_Params

        self.Actor  = Policy_NeuralNet(self.Actor_Hypers["Network Size"], self.State_Dim, self.Action_Dim,
                                       Learning_Rate = self.Actor_Hypers["Learning Rate"],
                                       Activation    = self.Actor_Hypers["Activation"],
                                       Epoch         = self.Actor_Hypers["Epoch"],
                                       Alpha         = self.Actor_Hypers["Alpha"],
                                       Batch_Size    = self.Actor_Hypers["Batch Size"])

        self.Critic = Critic_NeuralNet(self.Critic_Hypers["Network Size"], self.State_Dim, 1,
                                       Learning_Rate = self.Critic_Hypers["Learning Rate"],
                                       Activation    = self.Critic_Hypers["Activation"],
                                       Epoch         = self.Critic_Hypers["Epoch"],
                                       Alpha         = self.Critic_Hypers["Alpha"],
                                       Batch_Size    = self.Critic_Hypers["Batch Size"])

        self.Environment = Environment


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

        Exp = []
        Episode_Exps = []
        Step_Count = 0

        for i in tqdm.tqdm(range(N_Episodes)):
            Done = False
            Episode_Exp = []
            State_0 = self.Environment.reset()

            Sigma = max(self.Sigma_Range[0] - ((self.Sigma_Range[0] - self.Sigma_Range[1]) * (i / (self.Sigma_Anneal * N_Episodes))), self.Sigma_Range[1])
            Actor_LR = self.Actor_Hypers["Learning Rate"] * (Sigma)

            while Done == False:
                Mu = np.clip(self.Actor.Predict(State_0.reshape(1, self.State_Dim)), -10, 10)
                Leverage = np.random.normal(Mu, Sigma)

                State_1, Reward, Done, Info = self.Environment.step(Leverage[0])
                Episode_Exp.append({"s0" : State_0, "s1" : State_1, "r" : Reward, "a" : Leverage, 'i' : Info, 'Mu' : Mu.flatten()})
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


            if i % self.Retrain_Frequency == 0:
                State     = np.array([e['s0'] for e in Exp]).reshape((-1, self.State_Dim))
                Action    = np.array([e['a']  for e in Exp]).reshape((-1, self.Action_Dim))
                Reward    = np.array([e['r']  for e in Exp]).reshape((-1, 1))
                Advantage = Reward - self.Critic.Predict(State)

                self.Actor.Learning_Rate = Actor_LR
                self.Actor.Fit(State, Action, Advantage, Sigma)
                self.Critic.Fit(State, Reward)

                Exp = []


    def Predict_Action (self, X):
        ''' Returns the optimal action given a state '''
        return self.Actor.Predict(X.reshape(1, self.State_Dim))






























def Gen_Plots (Agent, Plot_Data):
    '''
    Generate and display plots

    Parameters
    ----------
        Agent     : A copy of the trained agent.
        Plot_Data : A dictionary with plot keys which contains the data to be plotted.

    Returns nothing but displays the charts.
    '''

    if len(Plot_Data.keys()) == 0 : return None

    N = len(Plot_Data.keys())
    if 'Merton_Sim' in Plot_Data.keys() : N += 1
    f, ax = plt.subplots(1,N, figsize = (6 * N, 6))
    i = 0

    if len(Plot_Data.keys()) == 1 : ax = [ax]


    if 'Mu' in Plot_Data.keys():
        ax[i].plot(np.array(Plot_Data['Mu']), label = 'Actions')
        ax[i].set_xlabel('Step')
        ax[i].set_ylabel('Leverage')
        ax[i].axhline(Agent.Environment.Training_Merton, color = 'k')
        i += 1

    if 'Mu_2' in Plot_Data.keys():
        ax[i].plot(np.array(Plot_Data['Mu_2']), label = 'Actions')
        ax[i].set_xlabel('Step')
        ax[i].set_ylabel('Leverage')
        ax[i].axhline(Agent.Environment.Training_Merton, color = 'k')
        i += 1

    if 'Merton_Sim' in Plot_Data.keys():
        assert isinstance(Agent.Environment.Mu, int) or isinstance(Agent.Environment.Mu, float), 'Merton_Sim only to be used with single asset.'

        for j in range(len(Plot_Data['Merton_Sim'])):
            ax[i].plot(np.linspace(0,1,20), Plot_Data['Merton_Sim'][j]["Policy"], label = Plot_Data['Merton_Sim'][j]["Title"])
            ax[i+1].plot(np.linspace(0,1,20), Plot_Data['Merton_Sim'][j]["Value"],  label = Plot_Data['Merton_Sim'][j]["Title"])

            ax[i].set_xlabel("Wealth")
            ax[i].set_ylabel("Leverage")
            ax[i].axhline(Agent.Environment.Training_Merton, color = 'k')
            ax[i].legend()

            ax[i+1].set_xlabel("Wealth")
            ax[i+1].set_ylabel("Utility")
            ax[i+1].legend()

            if Agent.Environment.Risk_Aversion == 1:
                ax[i+1].plot(np.linspace(0,1,20), [None] + list(np.log(np.linspace(0,1,20)[1::])), color = 'k')
            else:
                ax[i+1].plot(np.linspace(0,1,20), [None] + list((np.linspace(0,1,20)[1::] ** (1 - Agent.Environment.Risk_Aversion)) / (1 - Agent.Environment.Risk_Aversion)), color = 'k')
        i += 2

    if 'Greedy_Merton' in Plot_Data.keys():
        Plot_Data['Greedy_Merton'] = np.array(Plot_Data['Greedy_Merton'])
        ax[i].scatter(np.arange(Plot_Data['Greedy_Merton'].shape[0]), Plot_Data['Greedy_Merton'][:,0], label = 'AC', color = 'lightskyblue')
        ax[i].scatter(np.arange(Plot_Data['Greedy_Merton'].shape[0]), Plot_Data['Greedy_Merton'][:,1], label = 'Merton', color = 'mediumvioletred')
        ax[i].legend()
        i += 1

    if 'Percent_Merton_Action' in Plot_Data.keys():
        ax[i].scatter(np.arange(len(Plot_Data['Percent_Merton_Action'])), Plot_Data['Percent_Merton_Action'], color = 'mediumvioletred')
        i += 1



    plt.show()





# # Following functions calculate the merton fraction in a multi asset environment.
# def Calculate_Merton_Weights(Env):
#
#     if hasattr(Env, 'Training_Merton'):
#         return Env.Training_Merton
#
#     elif isinstance(Env.Mu, int) or isinstance(Env.Mu, float):
#         return (Env.Mu - Env.Rf) / (Env.Risk_Aversion * (Env.Sigma ** 2))
#
#     else:
#
#         cons = [{'type': 'ineq', 'fun': lambda x:  np.sum(x) - 1},
#                 {'type': 'ineq', 'fun': lambda x: -np.sum(x) + 1}]
#
#         Opt_Weights  = minimize(Sharpe_Ratio, [1 / len(Env.Mu)] * len(Env.Mu), args = (Env), constraints = cons).x
#         Opt_leverage = Merton_Leverage(Opt_Weights, Env)
#         Opt_Weights  = Opt_Weights * Opt_leverage
#
#         return Opt_Weights
#
# def Sharpe_Ratio(Weights, Env):
#     Weights = np.array(Weights)
#     Excess_ret = np.sum(Weights * Env.Mu) - Env.Rf
#
#     Std = Env.Sigma.reshape(-1,1)
#     Cov = Env.Row * np.matmul(Std, Std.T)
#
#     Weights = Weights.reshape(-1,1)
#     Var = np.matmul(np.matmul(Weights.T, Cov), Weights)
#
#     Sharpe = Excess_ret / (Var ** 0.5)[0,0]
#
#     return -Sharpe
#
# def Merton_Leverage(Weights, Env):
#     Weights = np.array(Weights)
#     Return = np.sum(Weights * Env.Mu)
#
#     Std = Env.Sigma.reshape(-1,1)
#     Cov = Env.Row * np.matmul(Std, Std.T)
#
#     Weights = Weights.reshape(-1,1)
#     Var = np.matmul(np.matmul(Weights.T, Cov), Weights)
#
#     return (Return - Env.Rf) / (Env.Risk_Aversion * (Var[0,0]))
