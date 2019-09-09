import numpy as np
import tensorflow as tf

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


class NN():
    def __init__(self, Name, Loss, Input_Dim, Output_Dim = 1, Hidden = [20, 10], Activation = 'Sigmoid', solver = 'Adam', Alpha = 0):
        self._Activation_Method = self._Activation(Activation)
        regularizer = tf.contrib.layers.l2_regularizer(Alpha)
        self.X = tf.placeholder(shape = [None, Input_Dim], dtype = tf.float32, name = 'Inputs')
        Hidden_Layers = self.X
        for layers in Hidden:
            Hidden_Layers = tf.layers.dense(Hidden_Layers, layers, activation= self._Activation_Method, activity_regularizer= regularizer)
        self.Predict = tf.layers.dense(Hidden_Layers, Output_Dim, activation= None, activity_regularizer= regularizer)

        self.learning_rate = tf.placeholder(tf.float32, shape = (), name = 'Learning_rate')
        self.Y = tf.placeholder(shape = [None, Output_Dim], dtype = tf.float32, name = 'Output')

        self._Optimizer = tf.train.AdamOptimizer(learning_rate= self.learning_rate)

        if Loss == 'l2':
            self.l = tf.losses.mean_squared_error(self.Y, self.Predict)

        if Loss == 'Gaussian':
            self.Adv = tf.placeholder(shape = [None, 1], dtype = tf.float32, name = 'Advantage')
            self.sigma = tf.placeholder(tf.float32, shape = (), name= 'Sigma')
            Loglik = tf.log(2*np.pi*self.sigma**2)/2 + 0.5*(self.Y - self.Predict)**2/(self.sigma**2)
            self.l = tf.matmul(tf.reshape(tf.reduce_sum(Loglik, axis = 1), shape= [1,-1]), self.Adv)

        self.Weights = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        self.fit = self._Optimizer.minimize(self.l)

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

        self.Actor  =  NN('Actor', 'Gaussian', self.State_Dim, self.Action_Dim,
                           Hidden      = self.Actor_Hypers["Network Size"],
                           Activation  = self.Actor_Hypers["Activation"],
                           Alpha       = self.Actor_Hypers['Alpha'])

        self.Critic =  NN('Critic', 'l2', self.State_Dim, 1,
                           Hidden       = self.Critic_Hypers["Network Size"],
                           Activation   = self.Critic_Hypers["Activation"],
                           Alpha        = self.Critic_Hypers['Alpha'])

        self.TF_Session   = tf.Session()
        self.TF_Session.run(tf.global_variables_initializer())

        self.Environment = Environment

    def Train (self, N_Episodes):

        Exp = []
        Mus = []

        Record_Eps  = np.linspace(0, N_Episodes, self.Plot_Frequency).astype(int) - 1
        Record_Eps[0] = 0
        Plot_Data = []

        for i in range(N_Episodes):
            Done = False
            Episode_Exp = []
            State_0 = self.Environment.reset()

            Sigma = max(self.Sigma_Range[0] - ((self.Sigma_Range[0] - self.Sigma_Range[1]) * (i / (self.Sigma_Anneal * N_Episodes))), self.Sigma_Range[1])
            Actor_LR = self.Actor_Hypers["Learning Rate"] * (Sigma)

            while Done == False:
                Mu = self.TF_Session.run(self.Actor.Predict, feed_dict = {self.Actor.X: State_0.reshape(-1, self.State_Dim)})
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
                V_hat     = self.TF_Session.run(self.Critic.Predict, feed_dict = {self.Critic.X: State})
                Advantage = Reward - V_hat

                for k in range(self.Actor_Hypers['Epoch']):
                    self.TF_Session.run(self.Actor.fit, feed_dict = {self.Actor.X:State, self.Actor.Y:Action, self.Actor.Adv: Advantage, self.Actor.sigma:Sigma, self.Actor.learning_rate:Actor_LR})

                for k in range(self.Critic_Hypers['Epoch']):
                    self.TF_Session.run(self.Critic.fit, feed_dict = {self.Critic.X:State, self.Critic.Y:Reward, self.Critic.learning_rate:self.Critic_Hypers['Learning Rate']})

                Exp = []

            if np.any(i == Record_Eps):
                Test_State = np.hstack((np.zeros((20,1)), np.linspace(0,1,20).reshape(-1,1)))
                if self.Action_Dim == 1:
                    Pol = self.TF_Session.run(self.Actor.Predict, feed_dict = {self.Actor.X : Test_State})
                    Val = self.TF_Session.run(self.Critic.Predict, feed_dict = {self.Critic.X : Test_State})
                    Plot_Data.append({"Policy" : Pol,
                                      "Value"  : Val,
                                      "Title"  : str(i + 1) + " Eps"})

#                else:
#                    Plot_Data.append({"Policy" : self.Actor.Predict(Test_State).reshape(-1, self.Action_Dim),
#                                      "Value"  : self.Critic.Predict(Test_State).reshape(-1),
#                                      "Title"  : str(i + 1) + " Eps"})

        return [Mus, Plot_Data]
