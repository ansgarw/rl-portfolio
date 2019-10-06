import numpy as np
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

from Neural_Network_Collections import Multihead_TF
from DQN_Template import DQN_Template


# The DQN Agent itself.
class DQN_MH_TF(DQN_Template):

    def __init__ (self, Environment, Action_Discretise, Network_Params, Gamma = 0.99, Epsilon_Range = [1, 0.1], Epsilon_Anneal = 0.5, Retrain_Frequency = 25):

        State_Dim = Environment.observation_space.shape[0]
        Model = Multihead_TF(State_Dim,   Action_Discretise,
                            Hidden        = Network_Params["Network Size"], 
                            Activation    = Network_Params["Activation"],
                            Learning_Rate = Network_Params["Learning Rate"],
                            Alpha         = Network_Params["Alpha"])

        super().__init__(Environment, Gamma, Epsilon_Range, Epsilon_Anneal, Model, Action_Discretise, Retrain_Frequency)
        
        self.Batch_Size        = Network_Params['Batch_Size']
        self.Epoch             = Network_Params['Epoch']
        self.TF_Session        = tf.Session()
        self.TF_Session.run(tf.global_variables_initializer())
        

    def Refit(self):
        if len(self.Exp) > self.Batch_Size*self.Epoch:
            Data = np.random.choice(self.Exp, size = (self.Epoch, self.Batch_Size), replace = False)
            for row in Data:
                X  = np.array([d['s0'] for d in row]).reshape((-1, self.State_Dim))
                Y  = np.array([d['r']  for d in row]).reshape((-1,1))
                S1 = np.array([d['s1'] for d in row]).reshape((-1, self.State_Dim))
                Done = np.array([(d['done'] == False) for d in row]).reshape(-1,1)
        
                # When generating the reward to fit towards we augment the immediate reward with the quality of the
                # subsequent state, assuming greedy action from that point on. This relationship is formalised by the
                # bellman equation. Q(s,a) = R + Argmax(a)(Q(s',a)) * Gamma
                S1_Val = self.Predict_Q(S1)*Done
                Y += S1_Val * self.Gamma
                self.TF_Session.run(self.Q_Network.fit, feed_dict={self.Q_Network.X: X, self.Q_Network.Q_In:Y})
        else:
            pass
        
    def Predict_Q (self, state):
        state = state.reshape(-1, self.State_Dim)
        ''' A simple pedict fucntion to extract predictions from the agent without having to call methods on the internal network '''
        return self.TF_Session.run(self.Q_Network.Q, feed_dict={self.Q_Network.X: state})

    def Choose_Action(self, state):
        state = state.reshape(-1, self.State_Dim)
        ''' Returns the optimal action per the Q Network '''
        return self.TF_Session.run(self.Q_Network.choose, feed_dict={self.Q_Network.X: state})
