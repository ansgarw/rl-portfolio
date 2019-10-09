import numpy      as np
import tensorflow as tf
from A2C_Template import A2C_Template

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
# tf.compat.v1.disable_v2_behavior()


# NOTE: tf.layers is deprecated, consider using keras.layers instead?
class AC_Network ():

    def __init__(self, Input_Dim, Action_Dim, Hidden_Layers, Predict_Sigma, Activation = 'Sigmoid', Solver = 'ADAM', Alpha_A = 0, Alpha_V = 0):

        self.Activation_Method = self.Activation(Activation)
        Actor_Reg  = tf.contrib.layers.l2_regularizer(Alpha_A) if Alpha_A != 0 else None
        Critic_Reg = tf.contrib.layers.l2_regularizer(Alpha_V) if Alpha_V != 0 else None

        # Create an input placeholder
        self.X = tf.placeholder(shape = [None, Input_Dim], dtype = tf.float32, name = 'States')

        # Next loop through all the required hidden layers, setting the input of each as the output of the last
        Last_Layer = self.X
        for Layer in Hidden_Layers:
            Last_Layer = tf.layers.dense(Last_Layer, Layer, activation = self.Activation_Method, activity_regularizer = Actor_Reg)

        # Finally create the output layers.
        self.Predict_Action = tf.layers.dense(Last_Layer, Action_Dim, activation = None,           activity_regularizer = Actor_Reg)
        self.Predict_Sigma  = tf.layers.dense(Last_Layer, Action_Dim, activation = tf.nn.softplus, activity_regularizer = Actor_Reg)
        self.Predict_Value  = tf.layers.dense(Last_Layer, 1,          activation = None,           activity_regularizer = Critic_Reg)

        self.Learning_Rate = tf.placeholder(tf.float32, shape = (), name = 'Learning_Rate')

        # Next create the inputs required for the BackProp process.
        self.Value     = tf.placeholder(shape = [None, 1], dtype = tf.float32, name = 'Value')
        self.Advantage = tf.placeholder(shape = [None, 1], dtype = tf.float32, name = 'Advantage')
        self.Action    = tf.placeholder(shape = [None, Action_Dim], dtype = tf.float32, name = 'Action')
        self.Sigma     = tf.placeholder(shape = [None, Action_Dim], dtype = tf.float32, name = 'Sigma')

        self.Loglik = (tf.log(2 * np.pi * self.Sigma ** 2) / 2) + (0.5 * (self.Action - self.Predict_Action) ** 2 / (self.Sigma ** 2))

        Critic_Loss = tf.losses.mean_squared_error(self.Value, self.Predict_Value)
        Actor_Loss  = tf.matmul(tf.reshape(tf.reduce_sum(self.Loglik, axis = 1), shape = [1,-1]), self.Advantage)


        if self.Predict_Sigma == True:
            Entropy = tf.reduce_sum(tf.log(2 * np.e * np.pi) ** 0.5 * self.Sigma)
            Loss = Actor_Loss + Critic_Loss - Entropy
        else:
            Loss = Actor_Loss + Critic_Loss


        if Solver == 'SGD':
            self.Optimizer = tf.train.GradientDescentOptimizer(learning_rate = self.Learning_Rate)
        else:
            self.Optimizer = tf.train.AdamOptimizer(learning_rate = self.Learning_Rate)

        self._Fit = self.Optimizer.minimize(Loss)


    def Activation(self, Method):
        ''' Specify the activation function from its keyword. '''

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




class Actor_Critic (A2C_Template) :

    def __init__ (self, Environment, Network_Params, Gamma, Sigma_Range, Sigma_Anneal, Predict_Sigma = False, Retrain_Frequency = 100):
        '''
        Set up the Actor Critic.
        '''

        self.Predict_Sigma     = Predict_Sigma
        self.Sigma_Range       = Sigma_Range
        self.Sigma_Anneal_Frac = Sigma_Anneal
        self.Network_Params    = Network_Params

        if self.Predict_Sigma == False:
            self.Sigma = self.Sigma_Range[0]
            self.Refit_Count = 0

        self.Network = AC_Network(Environment.observation_space.shape[0], Environment.action_space.shape[0], Network_Params['Network_Size'],
                             Predict_Sigma = self.Predict_Sigma,
                             Activation    = Network_Params['Activation'],
                             Solver        = Network_Params['Solver'],
                             Alpha_A       = Network_Params['Actor_Alpha'],
                             Alpha_V       = Network_Params['Critic_Alpha'])

        super().__init__(Environment, Gamma, Retrain_Frequency)

        self.TF_Session = tf.Session()
        self.TF_Session.run(tf.global_variables_initializer())


    def Predict_Action (self, State):
        ''' Returns: Mu, Sigma. '''

        Mu = self.TF_Session.run(self.Network.Predict_Action, feed_dict = {self.Network.X: State.reshape(-1, self.State_Dim)})

        if self.Predict_Sigma == True:
            Sigma = self.TF_Session.run(self.Network.Predict_Sigma,  feed_dict = {self.Network.X: State.reshape(-1, self.State_Dim)})
            Sigma = np.clip(Sigma, Sigma[1], Sigma[0])
        else:
            Sigma = np.array([self.Sigma])

        return Mu, Sigma


    def Train (self, N_Episodes, *args):
        self.Sigma_Anneal = self.Sigma_Anneal_Frac * int(N_Episodes / self.Retrain_Frequency)

        if len(args) == 1:
            super().Train(N_Episodes, Plot = args[0])
        else:
            super().Train(N_Episodes)


    # IDEA: Still need to account for minibatch in here.
    def Refit_Model (self, Exp):
        ''' Refit the network. '''

        State     = np.array([e['s0'] for e in Exp]).reshape((-1, self.State_Dim))
        Action    = np.array([e['a']  for e in Exp]).reshape((-1, self.Action_Dim))
        Reward    = np.array([e['r']  for e in Exp]).reshape((-1, 1))
        Advantage = Reward - self.TF_Session.run(self.Network.Predict_Value, feed_dict = {self.Network.X : State})

        if self.Predict_Sigma == False:
            # Decay sigma each time the model is refit.
            self.Refit_Count += 1
            self.Sigma = max(self.Sigma_Range[0] - ((self.Sigma_Range[0] - self.Sigma_Range[1]) * (self.Refit_Count / self.Sigma_Anneal)), self.Sigma_Range[1])
            Sigma = np.array([self.Sigma]).reshape(1,1)
            Learning_Rate = self.Network_Params['Learning_Rate'] * Sigma[0][0]

        else:
            Sigma = np.array([e['Sigma'] for e in Exp]).reshape((-1, self.Action_Dim))
            Learning_Rate = self.Network_Params['Learning_Rate']

        self.TF_Session.run(self.Network._Fit, feed_dict = {self.Network.X             : State,
                                                            self.Network.Action        : Action,
                                                            self.Network.Value         : Reward,
                                                            self.Network.Sigma         : Sigma,
                                                            self.Network.Advantage     : Advantage,
                                                            self.Network.Learning_Rate : Learning_Rate})
