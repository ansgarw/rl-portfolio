import numpy as np
import tensorflow as tf
from A2C_Template import A2C_Template
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


class Actor_Network():
    def __init__(self, Hidden, Input_Dim, Action_Dim,  Activation = 'Sigmoid',  Alpha = 0):
        
        self._Activation_Method = self._Activation(Activation)
        regularizer_A = tf.contrib.layers.l2_regularizer(Alpha) if Alpha!= 0 else None

        self.X = tf.placeholder(shape = [None, Input_Dim], dtype = tf.float32, name = 'States')   
        Hidden_Layers = self.X
        for layers in Hidden:
            Hidden_Layers = tf.layers.dense(Hidden_Layers, layers, activation= self._Activation_Method, activity_regularizer= regularizer_A)  
            
        self.Predict = tf.layers.dense(Hidden_Layers, Action_Dim, activation= None, activity_regularizer= regularizer_A)
        self.Sigma_Predict = tf.layers.dense(Hidden_Layers, Action_Dim, activation= tf.nn.softplus, activity_regularizer= regularizer_A)
        
        self.learning_rate = tf.placeholder(tf.float32, shape = (), name = 'Learning_rate')
        self._Optimizer = tf.train.AdamOptimizer(learning_rate= self.learning_rate)
            
        self.Adv = tf.placeholder(shape = [None, 1], dtype = tf.float32, name = 'Advantage')
        self.A_in = tf.placeholder(shape= [None, Action_Dim], dtype= tf.float32, name = 'Actions')
        self.sigma_in = tf.placeholder(tf.float32, shape = [None, Action_Dim], name= 'Sigma')
        self.Loglik = tf.log(2*np.pi*self.sigma_in**2)/2 + 0.5*(self.A_in - self.Predict)**2/(self.sigma_in**2)  

        Entropy = tf.reduce_sum(tf.log(2*np.e*np.pi)**0.5*self.sigma_in)
        P_Loss = tf.matmul(tf.reshape(tf.reduce_sum(self.Loglik, axis = 1), shape= [1,-1]), self.Adv)
        Loss = P_Loss - Entropy
        self.Weights = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        self.fit = self._Optimizer.minimize(Loss)

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



class Critic_Network():
    def __init__(self, Hidden, Input_Dim, Alpha_V = 0):
        
        self._Activation_Method = tf.nn.sigmoid
        regularizer_V = tf.contrib.layers.l2_regularizer(Alpha_V) if Alpha_V!= 0 else None

        self.X = tf.placeholder(shape = [None, Input_Dim], dtype = tf.float32, name = 'States')   
        Hidden_Layers = self.X
        for layers in Hidden:
            Hidden_Layers = tf.layers.dense(Hidden_Layers, layers, activation= self._Activation_Method, activity_regularizer= regularizer_V)  

        self.Value_Pred = tf.layers.dense(Hidden_Layers, 1, activation = None, activity_regularizer= regularizer_V)     
        self.learning_rate = tf.placeholder(tf.float32, shape = (), name = 'Learning_rate')
        
        self._Optimizer = tf.train.AdamOptimizer(learning_rate= self.learning_rate)
        self.V_in = tf.placeholder(shape = [None, 1], dtype = tf.float32, name = 'Output')
        V_Loss = tf.losses.mean_squared_error(self.V_in, self.Value_Pred)
            
        self.Weights = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        self.fit = self._Optimizer.minimize(V_Loss)
        
        
class Critic_Polynomial():
    def __init__(self, Input_Dim, power = 3):
        self.X = tf.placeholder(shape = [None, Input_Dim], dtype = tf.float32, name = 'States')   
        b = tf.Variable(tf.random_normal([1]), name = 'bias')
        Y = 0
        for i in range(power):
            x_pow = tf.pow(self.X, i+1)
            weights = tf.Variable(tf.random_normal([Input_Dim, 1]))
            Y += tf.matmul(x_pow, weights)
            
        self.Value_Pred = Y + b
        self.learning_rate = tf.placeholder(tf.float32, shape = (), name = 'Learning_rate')
        self._Optimizer = tf.train.AdamOptimizer(learning_rate= self.learning_rate)
        
        self.V_in = tf.placeholder(shape = [None, 1], dtype = tf.float32, name = 'Output')
        V_Loss = tf.losses.mean_squared_error(self.V_in, self.Value_Pred)
        self.Weights = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        self.fit = self._Optimizer.minimize(V_Loss)
            

def Empty (*args):
    ''' An empty function which accepts any number of arguments '''
    pass    

    

class Actor_Critic (A2C_Template):
    
    def __init__ (self, Environment, Actor_Hypers, Critic_Hypers, Gamma, Sigma_Range, Sigma_Anneal, Retrain_Frequency):

        super().__init__(Environment, Gamma, Retrain_Frequency)
        self.Sigma_Range       = Sigma_Range
        self.Sigma_Anneal      = Sigma_Anneal
        self.Lambda            = 0.95
                        
        self.Actor = Actor_Network(Actor_Hypers["Network Size"], self.State_Dim, self.Action_Dim,
                                   Activation    = Actor_Hypers["Activation"],
                                   Alpha         = Actor_Hypers["Alpha"])
        
        self.Critic = Critic_Network(Critic_Hypers["Network Size"], self.State_Dim,Critic_Hypers["Alpha"])        
#        self.Critic = Critic_Polynomial(self.State_Dim, power = Critic_Hypers['power'])        
        self.Epoch = Actor_Hypers['Epoch']
        self.Base_LR = Actor_Hypers['Learning Rate']
        self.TF_Session   = tf.Session()
        self.TF_Session.run(tf.global_variables_initializer())
        self._Current_Model = 0
        self.Sigma_Ceiling = Sigma_Range[0]
        

    def BackProp_Reward(self, Episode_Exp):
        S0 = np.array([e['s0'] for e in Episode_Exp]).reshape((-1, self.State_Dim))
        S1 = np.array([e['s1'] for e in Episode_Exp]).reshape((-1, self.State_Dim))
        mark = np.array([e['done'] for e in Episode_Exp]).reshape(-1, 1)
        R = np.array([e['r'] for e in Episode_Exp]).reshape(-1, 1)
        
        V_S0 = self.TF_Session.run(self.Critic.Value_Pred, feed_dict = {self.Critic.X: S0})
        V_S1 = self.TF_Session.run(self.Critic.Value_Pred, feed_dict = {self.Critic.X: S1})   
        Delta = V_S1*self.Gamma*(1-mark) + R - V_S0               
        GAE = []                
        gae = 0
        for d in Delta[::-1]:
            gae += d[0]*self.Gamma*self.Lambda
            GAE.append(gae)
        GAE = np.array(GAE[::-1]).reshape(-1,1)
        V_S0 += GAE              
        for i in range(len(Episode_Exp)):
            Episode_Exp[i].update({'r': V_S0[i], 'Advantage':GAE[i]})
            
        return Episode_Exp
    
    def Refit_Model (self, Experience):
        self.Learning_Rate = self.Base_LR * self.Sigma_Ceiling
        Exp = sum(Experience, [])
        S0 = np.array([e['s0'] for e in Exp]).reshape((-1, self.State_Dim))
        R = np.array([e['r'] for e in Exp]).reshape(-1, 1)
        Actions = np.array([e['a'] for e in Exp]).reshape(-1, self.Action_Dim)
        Sigmas = np.array([e['Sigma'] for e in Exp]).reshape(-1, self.Action_Dim) 
        Advantage = np.array([e['Advantage'] for e in Exp]).reshape(-1,1)
        
        n = S0.shape[0]
        ind = np.random.choice(n, size=(self.Epoch, n//self.Epoch), replace=False)
        
        for k in range(self.Epoch):
            self.TF_Session.run(self.Actor.fit, feed_dict = {self.Actor.X:S0[ind[k],:], self.Actor.A_in:Actions[ind[k],:], 
                            self.Actor.sigma_in: Sigmas[ind[k],:], self.Actor.Adv:Advantage[ind[k],:], self.Actor.learning_rate:self.Learning_Rate})    
    
            self.TF_Session.run(self.Critic.fit, feed_dict = {self.Critic.X:S0[ind[k],:], self.Critic.V_in: R[ind[k],:], self.Critic.learning_rate:self.Base_LR})    
            
        self._Current_Model += 1
        self.Sigma_Ceiling = max(self.Sigma_Range[0] - ((self.Sigma_Range[0] - self.Sigma_Range[1]) * (self._Current_Model / (self.Sigma_Anneal * self.N_Updates))), self.Sigma_Range[1])
        
        
    def Predict_Action(self, state, oos = False):
        if self._Current_Model == 0:
            Mu = np.zeros((1,self.Action_Dim))
            Sigma = np.ones((1,self.Action_Dim))*self.Sigma_Range[0]
            
        else:
            Mu = self.TF_Session.run(self.Actor.Predict, feed_dict = {self.Actor.X: state.reshape(-1, self.State_Dim)})
            Sigma = self.TF_Session.run(self.Actor.Sigma_Predict, feed_dict = {self.Actor.X: state.reshape(-1, self.State_Dim)})
            if oos == False:
                Sigma = np.clip(Sigma, self.Sigma_Range[1], self.Sigma_Ceiling)
            
        return Mu, Sigma
    

    
    