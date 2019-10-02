import tqdm
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


class NN():
    def __init__(self, Name, Loss, Input_Dim, Output_Dim = 1, Hidden = [20, 10], Activation = 'Sigmoid', solver = 'Adam', Alpha = 0):
        self._Activation_Method = self._Activation(Activation)
        regularizer = tf.contrib.layers.l2_regularizer(Alpha) if Alpha!= 0 else None
        self.X = tf.placeholder(shape = [None, Input_Dim], dtype = tf.float32, name = 'States')
        Hidden_Layers = self.X
        for layers in Hidden:
            Hidden_Layers = tf.layers.dense(Hidden_Layers, layers, activation= self._Activation_Method, activity_regularizer= regularizer)
        self.Q = tf.layers.dense(Hidden_Layers, Output_Dim, activation= None, activity_regularizer= regularizer)

        self.learning_rate = tf.placeholder(tf.float32, shape = (), name = 'Learning_rate')
        self.Q_In = tf.placeholder(shape = [None, Output_Dim], dtype = tf.float32, name = 'Quality')

        self._Optimizer = tf.train.AdamOptimizer(learning_rate= self.learning_rate)
        self.loss = tf.losses.mean_squared_error(self.Q_In, self.Q)

        self.Weights = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        self.fit = self._Optimizer.minimize(self.loss)

    def _Activation(self, how):
        if how == 'Sigmoid':
            A_fun = tf.nn.sigmoid
        elif how == 'Relu':
            A_fun = tf.nn.relu
        elif how == 'Tanh':
            A_fun = tf.nn.tanh
        elif how == 'Softplus':
            A_fun = tf.nn.softplus
        return A_fun


# The DQN Agent itself.
class DQN:

    def __init__ (self, Environment, Action_Dim, Network_Params, Gamma = 0.99, Batch_Size = 512, Epsilon_Range = [1, 0.1], Epsilon_Anneal = 0.1, Retrain_Frequency = 100):
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



        self.Q_Network = NeuralNet(self.Network_Params["Network Size"], self.State_Dim, self.Action_Dim,
                                   Learning_Rate = self.Network_Params["Learning Rate"],
                                   Activation    = self.Network_Params["Activation"],
                                   Epoch         = self.Network_Params["Epoch"],
                                   Alpha         = self.Network_Params["Alpha"],
                                   Batch_Size    = self.Network_Params["Batch_Size"])

        self.Exp = []


    def Train (self, N_Episodes, Plot = []):

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

        Plot_Data = {}
        for key in Plot:
            Plot_Data[key] = []

        epsilons = self.Epsilon_Range[0] * np.ones(N_Episodes) - (1 / self.Epsilon_Anneal) * np.arange(0, N_Episodes) * (self.Epsilon_Range[0] - self.Epsilon_Range[1])
        epsilons = np.maximum(epsilons, self.Epsilon_Range[1])


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
                Episode_Exp.append({"s0" : State_0, "s1" : State_1, "r" : Reward, "a" : Action_idx, "i" : Info, "done" : Done})
                State_0 = State_1

            self.Exp.extend(Episode_Exp)
            if len(self.Exp) > 1e6:
                self.Exp[0:len(Episode_Exp)] = []

            # Plot per episode metrics
            if 'Merton_Benchmark' in Plot : Plot_Data['Merton_Benchmark'].append(self.Merton_Benchmark(Episode_Exp))


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

                if 'Ave_Perf' in Plot : Plot_Data['Ave_Perf'].append(self.Average_Performance_Plot())
                if 'Greedy_Merton' in Plot : Plot_Data['Greedy_Merton'].append(self.Merton_Benchmark_Two())

        return Plot_Data


    def Predict_Q (self, X):
        ''' A simple pedict fucntion to extract predictions from the agent without having to call methods on the internal network '''
        return self.Q_Network.Predict(X)


    def Predict_Action(self, X):
        ''' Returns the optimal action per the Q Network '''
        return self.Action_Space[np.argmax(self.Q_Network.Predict(X), axis = 1)]


    # Plotting Functions
    def Merton_Benchmark (self, Episode_Exp):
        '''
        Returns the difference between the Utility of the agent and the utility of
        investing in the merton portfolio calculated across the training dataset.
        '''

        # Inital wealth is the first entry to the first state
        Intial_Wealth = Episode_Exp[0]['s0'][0]
        Merton_Return = Intial_Wealth

        for i in range(len(Episode_Exp)):
            Merton_Return *= (1 + Episode_Exp[i]['i']['Rfree'] + Episode_Exp[i]['i']['Mkt-Rf'] * self.Environment.Training_Merton)

        return Episode_Exp[-1]['r'] - self.Environment.Utility(Merton_Return)


    def Merton_Benchmark_Two (self):
        '''
        Assess the performance of N episodes against the merton portfolio, acting optimally.

        Returns
        -------
        A list of two floats:
            0. The Average utility across 100 episodes of the DQN
            1. The Average utility across 100 episodes of the Merton Portfolio

        '''
        results = {'DQN'    : [],
                   'Merton' : []}

        for i in range(100):
            Done = False
            Merton_Wealth = 1.0
            State = self.Environment.reset()
            self.Environment.Wealth = 1

            while Done == False:
                Action_idx = np.argmax(self.Q_Network.Predict(State.reshape(1, self.State_Dim)))
                State, Reward, Done, Info = self.Environment.step(self.Action_Space[Action_idx])
                Merton_Wealth *= (1 + Info['Rfree'] + Info['Mkt-Rf'] * self.Environment.Training_Merton)

            results['DQN'].append(Reward)
            results['Merton'].append(self.Environment.Utility(Merton_Wealth))

        return [np.mean(results['DQN']), np.mean(results['Merton'])]


    def Average_Performance_Plot (self):
        ''' Run through 100 episodes acting greedily and report the averaged terminal reward '''
        Rewards = []

        for i in range(100):
            Done = False
            State = self.Environment.reset()
            self.Environment.Wealth = 1

            while Done == False:
                Action_idx = np.argmax(self.Q_Network.Predict(State.reshape(1, self.State_Dim)))
                State, Reward, Done, Info = self.Environment.step(self.Action_Space[Action_idx])

            Rewards.append(Reward)

        return np.mean(Rewards)
