import numpy                as np
import tensorflow.compat.v1 as tf
import tqdm
import warnings

tf.disable_v2_behavior()
tf.logging.set_verbosity(tf.logging.ERROR)

def Empty (*args):
    ''' An empty function which accepts any number of arguments '''
    pass



class Policy_Network():

    def __init__(self, Shape, Input_Dim, Output_Dim, Alpha = 0, Activation = 'Sigmoid'):


        Activation_Method = self._Activation(Activation)
        Regularizer       = tf.keras.regularizers.l2(Alpha) if Alpha != 0 else None

        self.State     = tf.placeholder(shape = [None, Input_Dim], dtype = tf.float32, name = 'A_States')
        self.Action    = tf.placeholder(shape = [None, Output_Dim], dtype = tf.float32, name = 'Action_True')
        self.Advantage = tf.placeholder(shape = [None, 1], dtype = tf.float32, name = 'Advantage')
        self.Sigma     = tf.placeholder(shape = [None, Output_Dim], dtype = tf.float32, name = 'Sigma')

        Last_Layer = self.State
        for Layer in Shape:
            Last_Layer = tf.layers.dense(Last_Layer, Layer, activation = Activation_Method, activity_regularizer = Regularizer)

        self.Action_Pred = tf.layers.dense(Last_Layer, Output_Dim, activation = None, activity_regularizer = Regularizer, name = 'Action_Pred')

        self.Learning_Rate = tf.placeholder(tf.float32, shape = (), name = 'A_Learning_Rate')

        self.Optimizer = tf.train.AdamOptimizer(learning_rate = self.Learning_Rate)

        Loglik_Loss = tf.log(tf.sqrt(2 * np.pi * (self.Sigma ** 2))) + ((self.Action - self.Action_Pred) ** 2) / (2 * (self.Sigma ** 2))
        Loglik_Loss = tf.reduce_sum(Loglik_Loss, axis = 1, keepdims = True) * self.Advantage

        self.Fit = self.Optimizer.minimize(tf.reduce_mean(Loglik_Loss))

    def _Activation(self, Method):

        '''
        Parameters
        ----------
            Method | string
                The keyword of the activation function to use. Acceptable inputs include ['Sigmoid', 'Relu', 'Tanh', 'Softplus']
        '''

        if Method == 'Sigmoid':
            return tf.nn.sigmoid

        elif Method == 'Relu':
            return tf.nn.relu

        elif Method == 'Tanh':
            return tf.nn.tanh

        elif Method == 'Softplus':
            return tf.nn.Softplus

        else:
            warnings.warn('Activation function ' + str(Method) + ' not recognised, defaulting to Sigmoid')
            return tf.nn.sigmoid




class Value_Network():

    def __init__(self, Shape, Input_Dim, Output_Dim, Alpha = 0, Activation = "Sigmoid`"):

        Activation_Method = self._Activation(Activation)
        Regularizer       = tf.keras.regularizers.l2(Alpha) if Alpha != 0 else None

        self.State = tf.placeholder(shape = [None, Input_Dim], dtype = tf.float32, name = 'C_States')
        self.Value = tf.placeholder(shape = [None, Output_Dim], dtype = tf.float32, name = 'Value_True')

        Last_Layer = self.State
        for Layer in Shape:
            Last_Layer = tf.layers.dense(Last_Layer, Layer, activation = Activation_Method, activity_regularizer = Regularizer)

        self.Value_Pred = tf.layers.dense(Last_Layer, Output_Dim, activation = None, activity_regularizer = Regularizer, name = 'Value_Pred')

        self.Learning_Rate = tf.placeholder(tf.float32, shape = (), name = 'C_Learning_Rate')

        self.Optimizer = tf.train.AdamOptimizer(learning_rate = self.Learning_Rate)

        self.Fit = self.Optimizer.minimize(tf.losses.mean_squared_error(self.Value, self.Value_Pred))

    def _Activation(self, Method):

        '''
        Parameters
        ----------
            Method | string
                The keyword of the activation function to use. Acceptable inputs include ['Sigmoid', 'Relu', 'Tanh', 'Softplus']
        '''

        if Method == 'Sigmoid':
            return tf.nn.sigmoid

        elif Method == 'Relu':
            return tf.nn.relu

        elif Method == 'Tanh':
            return tf.nn.tanh

        elif Method == 'Softplus':
            return tf.nn.Softplus

        else:
            warnings.warn('Activation function ' + str(Method) + ' not recognised, defaulting to Sigmoid')
            return tf.nn.sigmoid




class TD_Lambda_Engine:

    def __init__(self, Gamma):

        self.Gamma = Gamma
        self.Extend_Gamma(101)

    def Extend_Gamma (self, Lenght):

        self.Gammas = np.ones(101) * self.Gamma
        for i in range(0, self.Gammas.size):
            self.Gammas[i] = self.Gammas[i] ** i

    def TD_Lambda (self, Rewards, Values, Lambda):

        if Rewards.size > self.Gammas.size:
            self.Extend_Gamma(Rewards.size + 1)

        TD_Returns = np.zeros(Rewards.size)
        for i in range(len(Rewards)):
            TD_Returns[i] = np.sum(Rewards[:i+1] * self.Gammas[:i+1]) + Values[i] * self.Gammas[i+1]

        Value = 0
        Area  = 0
        for i in range(len(TD_Returns) - 1):
            Value += (1 - Lambda) * (Lambda ** i) * TD_Returns[i]
            Area += (1 - Lambda) * (Lambda ** i)

        Value += (1 - Area) * TD_Returns[-1]

        return Value




class Actor_Critic:

    def __init__ (self, Environment, Actor_Hypers, Critic_Hypers, Gamma, Sigma_Range, Sigma_Anneal, Retrain_Frequency, Action_Space_Clip = 10, Experiance_Mode = 'Monte_Carlo', TD_Lambda = [1, 0.8, 0.5], Monte_Carlo_Frac = 0.1, Ignore_Actor_Frac = 0.0):

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

            Action_Space_Clip | float
                The value at which to clip the levergae the agent can take, to prevent it from randomly acting too agressively.

            Experiance_Mode | string
                A key which indicates the method to be used to generate experiance targets. Acceptable inputs include: ['Monte_Carlo', 'TD_1', 'TD_Lambda']

            TD_Lambda | float, list
                The lambda to use if using Experiance_Mode 'TD_Lambda'. If a float is passes lambda is constant. If a list of length 3 is passed then Lambda will fall from the 0th value to the 1st value, and will take the 2nd value fraction of training episodes to do so. i.e. by default will fall from 1 to 0.8 across 0.5 of the training episodes. A value of 1 is equivalent to Monte_Carlo, and a value of zero is equivalent to TD_1.

            Monte_Carlo_Frac | float
                The fraction of episodes to run overwriting Experiance_Mode with 'Monte_Carlo', as this method is most stable when the critic is poorly trained at the start of the training sequence.

            Ignore_Actor_Frac | float
                The fraction of episodes to train only Critic. Prevents the Actor being trained on nonsense at the start of training.


            Actor_Hypers | dict
                'Network Size'  | int         | The size of the Actor NN
                'Learning Rate' | float, list | The learning rate of the Actor.
                'Activation'    | string      | The activation of the Actor. Acceptable inputs include ['Relu', 'Sigmoid', 'Tanh', 'Softplus']
                'Epoch'         | int         | The Actor Epoch
                'Alpha'         | float       | L2 regularization coefficient for the Actor.

            Critic_Hypers | dict
                'Network Size'  | int         | The size of the Critic NN
                'Learning Rate' | float, list | The learning rate of the Critic.
                'Activation'    | string      | The activation of the Critic. Acceptable inputs include ['Relu', 'Sigmoid', 'Tanh', 'Softplus']
                'Epoch'         | int         | The Critic Epoch
                'Alpha'         | float       | L2 regularization coefficient for the Critic.

        '''

        self.Retrain_Frequency = Retrain_Frequency
        self.Sigma_Range       = Sigma_Range
        self.Sigma_Anneal      = Sigma_Anneal
        self.Gamma             = Gamma
        self.State_Dim         = Environment.observation_space.shape[0]
        self.Action_Dim        = Environment.action_space.shape[0]
        self.Action_Space_Clip = Action_Space_Clip

        self.Experiance_Mode   = Experiance_Mode
        self.TD_Lambda         = TD_Lambda
        self.Monte_Carlo_Frac  = Monte_Carlo_Frac
        self.Ignore_Actor_Frac = Ignore_Actor_Frac

        assert self.Experiance_Mode in ['Monte_Carlo', 'TD_1', 'TD_Lambda'], 'Experiance_Mode must be one of [Monte_Carlo, TD_1, TD_Lambda]'

        self.Actor_Hypers  = Actor_Hypers
        self.Critic_Hypers = Critic_Hypers

        self.TD_Lambda_Eng = TD_Lambda_Engine(self.Gamma)

        self.Actor  = Policy_Network(self.Actor_Hypers["Network Size"], self.State_Dim, self.Action_Dim,
                                     Activation    = self.Actor_Hypers["Activation"],
                                     Alpha         = self.Actor_Hypers["Alpha"])

        self.Critic = Value_Network(self.Critic_Hypers["Network Size"], self.State_Dim, 1,
                                    Activation    = self.Critic_Hypers["Activation"],
                                    Alpha         = self.Critic_Hypers["Alpha"])

        self.Environment = Environment

        self.TF_Session = tf.Session()
        self.TF_Session.run(tf.global_variables_initializer())


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

        Episode_Exps = []
        Plot_Exps    = []
        Step_Count = 0

        for i in tqdm.tqdm(range(N_Episodes)):
            Done = False
            Episode_Exp = []
            State_0 = self.Environment.reset()

            Sigma = max(self.Sigma_Range[0] - ((self.Sigma_Range[0] - self.Sigma_Range[1]) * (i / (self.Sigma_Anneal * N_Episodes))), self.Sigma_Range[1])

            while Done == False:
                # Mu = np.clip(self.Actor.Predict(State_0.reshape(1, self.State_Dim)), -self.Action_Space_Clip, self.Action_Space_Clip)
                Mu = self.Predict_Action(State_0).flatten()
                Leverage = np.random.normal(Mu, Sigma)

                State_1, Reward, Done, Info = self.Environment.step(Leverage)
                Episode_Exp.append({"s0" : State_0, "s1" : State_1, "r" : Reward, "a" : Leverage, 'i' : Info, 'Mu' : Mu, 'Sigma' : Sigma, 'd' : Done})
                State_0 = State_1
                Step_Count += 1


            Episode_Exps.append(Episode_Exp)
            Plot_Exps.append(Episode_Exp)

            if Step_Count > 10000:
                Plot(Plot_Exps)
                Plot_Exps  = []
                Step_Count = 0

            if i % int(N_Episodes / 5) == 0:
                Diag()

            if i % self.Retrain_Frequency == 0:
                _State, _Value, _Action, _Advantage, _Sigma = self.Gen_Experiance(Episode_Exps, (i / N_Episodes))
                Episode_Exps = []

                if (i / N_Episodes) > self.Ignore_Actor_Frac:
                    self.Fit_Actor(_State, _Action, _Advantage, _Sigma, Sigma)
                self.Fit_Critic(_State, _Value)


    def Fit_Actor (self, State, Action, Advantage, Sigma, LR_Mult):
        for _ in range(self.Actor_Hypers['Epoch']):

            if self.Actor_Hypers['Batch Size'] >= State.shape[0] or self.Actor_Hypers['Batch Size'] == 0:
                self.TF_Session.run(self.Actor.Fit, feed_dict = {self.Actor.State          : State,
                                                                  self.Actor.Action        : Action,
                                                                  self.Actor.Advantage     : Advantage,
                                                                  self.Actor.Sigma         : Sigma,
                                                                  self.Actor.Learning_Rate : self.Actor_Hypers['Learning Rate'] * LR_Mult})

            else:
                idx = np.random.choice(State.shape[0], size = (State.shape[0] // self.Actor_Hypers['Batch Size'], self.Actor_Hypers['Batch Size']), replace = False)
                for i in range(idx.shape[0]):
                    Sigma_ = Sigma[idx[i]] if hasattr(Sigma, 'shape') else Sigma
                    self.TF_Session.run(self.Actor.Fit, feed_dict = {self.Actor.State          : State[idx[i]],
                                                                      self.Actor.Action        : Action[idx[i]],
                                                                      self.Actor.Advantage     : Advantage[idx[i]],
                                                                      self.Actor.Sigma         : Sigma_,
                                                                      self.Actor.Learning_Rate : self.Actor_Hypers['Learning Rate'] * LR_Mult})


    def Fit_Critic (self, State, Value):
        for _ in range(self.Critic_Hypers['Epoch']):

            if self.Critic_Hypers['Batch Size'] >= State.shape[0] or self.Critic_Hypers['Batch Size'] == 0:
                self.TF_Session.run(self.Critic.Fit, feed_dict = {self.Critic.State         : State,
                                                                  self.Critic.Value         : Value,
                                                                  self.Critic.Learning_Rate : self.Critic_Hypers['Learning Rate']})

            else:
                idx = np.random.choice(State.shape[0], size = (State.shape[0] // self.Critic_Hypers['Batch Size'], self.Critic_Hypers['Batch Size']), replace = False)
                for i in range(idx.shape[0]):
                    self.TF_Session.run(self.Critic.Fit, feed_dict = {self.Critic.State         : State[idx[i]],
                                                                      self.Critic.Value         : Value[idx[i]],
                                                                      self.Critic.Learning_Rate : self.Critic_Hypers['Learning Rate']})


    def Gen_Experiance (self, Episode_Exps, Episode_Frac):
        '''
        Experiance may be generated via one of two methods:
            1. Monte Carlo : The terminal reward of an episode is discounted across the states visited in the epsiode
            2. Bellman     : The value of each state is deteremined by the immediate reward recieved and the discounted value of the next state, a relationship outlined by the bellman equation.

        The agent will dynamically switch between these two paradimes, using Monte Carlo to begin, and switching to bellman once the critic's loss has fallen sufficently.

        '''

        if self.Experiance_Mode == 'Monte_Carlo' or Episode_Frac < self.Monte_Carlo_Frac:
            for Episode_Exp in Episode_Exps:
                for j in range(len(Episode_Exp) - 1)[::-1]:
                    Episode_Exp[j]["r"] += Episode_Exp[j+1]["r"] * self.Gamma

            Exp = []
            for Episode_Exp in Episode_Exps:
                Exp.extend(Episode_Exp)

            State     = np.array([e['s0'] for e in Exp]).reshape((-1, self.State_Dim))
            Action    = np.array([e['a']  for e in Exp]).reshape((-1, self.Action_Dim))
            Value     = np.array([e['r']  for e in Exp]).reshape((-1, 1))
            Sigma     = np.array([e['Sigma'] for e in Exp]).reshape((-1, 1))
            Advantage = Value - self.Predict_Value(State)

            return State, Value, Action, Advantage, Sigma


        elif self.Experiance_Mode == 'TD_1':
            Exp = []
            for Episode_Exp in Episode_Exps:
                Exp.extend(Episode_Exp)

            State_0   = np.array([e['s0'] for e in Exp]).reshape((-1, self.State_Dim))
            State_1   = np.array([e['s1'] for e in Exp]).reshape((-1, self.State_Dim))
            Action    = np.array([e['a']  for e in Exp]).reshape((-1, self.Action_Dim))
            Reward    = np.array([e['r']  for e in Exp]).reshape((-1, 1))
            Done      = (np.array([e['d']  for e in Exp]) == False).reshape((-1, 1)).astype(int)
            Sigma     = np.array([e['Sigma'] for e in Exp]).reshape((-1, 1))

            Value     = Reward + self.Gamma * self.Predict_Value(State_1) * Done
            Advantage = Value - self.Predict_Value(State_0)

            return State_0, Value, Action, Advantage, Sigma


        elif self.Experiance_Mode == 'TD_Lambda':

            if isinstance(self.TD_Lambda, list):
                Lambda = max(self.TD_Lambda[0] - (Episode_Frac / self.TD_Lambda[2]) * (self.TD_Lambda[0] - self.TD_Lambda[1]), self.TD_Lambda[1])
            else:
                Lambda = self.TD_Lambda

            # Next we have to calculate the TD_Lambda value for each step in each episode...
            for Episode_Exp in Episode_Exps:
                V1 = self.Predict_Value(np.array([e['s1'] for e in Episode_Exp]).reshape((-1, self.State_Dim))).flatten()
                V1[-1] = 0
                R = np.array([e['r']  for e in Episode_Exp]).flatten()

                for i in range(len(Episode_Exp)):
                    Episode_Exp[i]['Value'] = self.TD_Lambda_Eng.TD_Lambda(R[i:], V1[i:], Lambda)

            Exp = []
            for Episode_Exp in Episode_Exps:
                Exp.extend(Episode_Exp)

            State  = np.array([e['s0'] for e in Exp]).reshape((-1, self.State_Dim))
            Value  = np.array([e['Value'] for e in Exp]).reshape((-1, 1))
            Action = np.array([e['a']  for e in Exp]).reshape((-1, self.Action_Dim))
            Sigma  = np.array([e['Sigma'] for e in Exp]).reshape((-1, 1))
            Advantage = Value - self.Predict_Value(State)

            return State, Value, Action, Advantage, Sigma


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
        X = X.reshape(-1, self.State_Dim)
        Mu = self.TF_Session.run(self.Actor.Action_Pred, feed_dict = {self.Actor.State : X})
        return np.clip(Mu, -self.Action_Space_Clip, self.Action_Space_Clip)


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
        X = X.reshape(-1, self.State_Dim)
        return self.TF_Session.run(self.Critic.Value_Pred, feed_dict = {self.Critic.State : X})
