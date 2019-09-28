import numpy as np
import time
import tensorflow as tf
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


class NN():
    def __init__(self, Name, Input_Dim, Action_Dim, Hidden, Activation = 'Sigmoid', solver = 'Adam', Alpha_A = 0, Alpha_V = 0):
        
        self._Activation_Method = self._Activation(Activation)
        regularizer_A = tf.contrib.layers.l2_regularizer(Alpha_A) if Alpha_A!= 0 else None
        regularizer_V = tf.contrib.layers.l2_regularizer(Alpha_V) if Alpha_V!= 0 else None

        self.X = tf.placeholder(shape = [None, Input_Dim], dtype = tf.float32, name = 'States')   
        Hidden_Layers = self.X
        for layers in Hidden:
            Hidden_Layers = tf.layers.dense(Hidden_Layers, layers, activation= self._Activation_Method, activity_regularizer= regularizer_A)  
            
        self.Predict = tf.layers.dense(Hidden_Layers, Action_Dim, activation= tf.nn.tanh, activity_regularizer= regularizer_A)
        self.Sigma_Predict = tf.layers.dense(Hidden_Layers, Action_Dim, activation= tf.nn.softplus, activity_regularizer= regularizer_A)
        self.Value_Pred = tf.layers.dense(Hidden_Layers, 1, activation = None, activity_regularizer= regularizer_V)
        
        self.learning_rate = tf.placeholder(tf.float32, shape = (), name = 'Learning_rate')
        
        if solver == 'SGD':
            self._Optimizer = tf.train.GradientDescentOptimizer(learning_rate= self.learning_rate)
        else:
            self._Optimizer = tf.train.AdamOptimizer(learning_rate= self.learning_rate)
 
        self.V_in = tf.placeholder(shape = [None, 1], dtype = tf.float32, name = 'Output')
        V_Loss = tf.losses.mean_squared_error(self.V_in, self.Value_Pred)
            
        self.Adv = tf.placeholder(shape = [None, 1], dtype = tf.float32, name = 'Advantage')
        self.A_in = tf.placeholder(shape= [None, Action_Dim], dtype= tf.float32, name = 'Actions')
        self.sigma_in = tf.placeholder(tf.float32, shape = [None, Action_Dim], name= 'Sigma')
        self.Loglik = tf.log(2*np.pi*self.sigma_in**2)/2 + 0.5*(self.A_in - self.Predict)**2/(self.sigma_in**2)  

        Entropy = tf.reduce_sum(tf.log(2*np.e*np.pi)**0.5*self.sigma_in)
        P_Loss = tf.matmul(tf.reshape(tf.reduce_sum(self.Loglik, axis = 1), shape= [1,-1]), self.Adv)
        Loss = V_Loss + P_Loss - Entropy
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
    

class Actor_Critic:
    
    def __init__ (self, Environment, AC_Params, Gamma, Sigma_Range, Sigma_Anneal, Retrain_Frequency):

        self.Retrain_Frequency = Retrain_Frequency
        self.Sigma_Range       = Sigma_Range
        self.Sigma_Anneal      = Sigma_Anneal
        self.Gamma             = Gamma
        self.Lambda            = 0.95
        self.State_Dim         = len(Environment.reset())
        self.Action_Dim        = len(Environment.action_space.sample())
        self.Plot_Frequency    = 10
                
        self.AC_Net  =  NN('AC', self.State_Dim, self.Action_Dim, 
                           Hidden      = AC_Params["Network Size"],
                           Activation  = AC_Params["Activation"],
                           solver      = AC_Params['solver'],
                           Alpha_A     = AC_Params['Alpha_A'],
                           Alpha_V     = AC_Params['Alpha_V'])
        
        self.Epoch = AC_Params['Epoch']
        self.Base_LR = AC_Params['Learning Rate']
        self.TF_Session   = tf.Session()
        self.TF_Session.run(tf.global_variables_initializer())
        
        self.Environment = Environment


    def Train (self, N_Updates):
        start = time.time()     # Timer to check how long each update takes
        Exp = {}                # Save Experience in a Dictionary
        Episode_Rewards = []    # Each Episode's experience used to calculate GAE and V_estimates
        i = 0                   # Count how many updates have been done
        fail = 0                # To monitor frequency of our failed experience before we obtain a success experience
        Success_count = 0       # Every Update we require certain amount of success experiences
        Fail_count = 0          # sum of fail to provide an overview of how each updated model perform
        
        while i < N_Updates:
            Done = False
            Episode_Exp = []
            State_0 = self.Environment.reset()
            Sigma_Ceiling = max(self.Sigma_Range[0] - ((self.Sigma_Range[0] - self.Sigma_Range[1]) * (i / (self.Sigma_Anneal * N_Updates))), self.Sigma_Range[1])
            Sigma_Floor = max(Sigma_Ceiling/2, self.Sigma_Range[1])
            self.Learning_Rate = self.Base_LR * Sigma_Ceiling
#            self.Sigma = Sigma_Ceiling
            local_R = 0
            while Done == False:
                if i ==0:
                    Leverage = np.random.normal([0]*self.Action_Dim, 1).reshape((1,1))
                    s = np.array(self.Sigma_Range[0]).reshape((1,1))
                else:
                    Mu = self.TF_Session.run(self.AC_Net.Predict, feed_dict = {self.AC_Net.X: State_0.reshape(-1, self.State_Dim)})
                    s = self.TF_Session.run(self.AC_Net.Sigma_Predict, feed_dict = {self.AC_Net.X: State_0.reshape(-1, self.State_Dim)})
                    
                    Mu = np.clip(Mu, -1, 1)
                    s = np.clip(s, Sigma_Floor, Sigma_Ceiling)
                    Leverage = np.random.normal(Mu, s)
                try:
                    State_1, Reward, Done, Info = self.Environment.step(Leverage[0])
                except:
                    raise Exception('Invalid Leverage {} and Mu {}'.format(Leverage[0], Mu))

                Episode_Exp.append([State_0, Leverage, Reward, Done, State_1, s])
                State_0 = State_1
                local_R += Reward

            if Reward >0:
                Exp = self.Record_Exp(Episode_Exp, Exp)
                Success_count += 1
                if Success_count % self.Plot_Frequency == 0 and Success_count!=0:
                    print('Episode {} of Model Version {} has reward {} after {} fails'.format(Success_count, i, local_R, Fail_count))
                    Fail_count = 0
                    Episode_Rewards.append(local_R)
                    
                Fail_count += fail
                fail = 0
                
                if Success_count == self.Retrain_Frequency:
                    print('Updating to Model Version {}...'.format(i+1))
                    Success_count  = 0
                    i+= 1        
                    self.Fit_Model(self.Epoch, Exp)
                    Exp = {}
                    t = time.time() - start
                    print('Update to Version {} has taken {} seconds'.format(i, t))
                    start = time.time()
                    
                    if i % 2 == 0 and i!=0:
                        self.Print_V_Function()
                        self.Print_Action_Space()
            else:
                if fail < 1:
                    Exp = self.Record_Exp(Episode_Exp, Exp)
                fail += 1

        return Episode_Rewards
    
    
    def Record_Exp(self, Episode_Exp, Rec_dictionary):
        temp = np.array(Episode_Exp)
        S0 = np.concatenate(temp[:, 0], axis = 0).reshape(-1, self.State_Dim)
        S1 = np.concatenate(temp[:, 4], axis = 0).reshape(-1, self.State_Dim)
        mark = temp[:,3].reshape(-1, 1)
        R = temp[:,2].reshape(-1,1)
        Actions = np.concatenate(temp[:, 1], axis = 0).reshape(-1, self.Action_Dim)
        Sigmas = np.concatenate(temp[:, 5], axis = 0).reshape(-1, self.Action_Dim) 
        
        V_S0 = self.TF_Session.run(self.AC_Net.Value_Pred, feed_dict = {self.AC_Net.X: S0})
        V_S1 = self.TF_Session.run(self.AC_Net.Value_Pred, feed_dict = {self.AC_Net.X: S1})   
        Delta = V_S1*self.Gamma*(1-mark) + R - V_S0               
        GAE = []                
        gae = 0
        for d in Delta[::-1]:
            gae += d[0]*self.Gamma*self.Lambda
            GAE.append(gae)
        GAE = np.array(GAE[::-1]).reshape(-1,1)
        V_S0 += GAE              
        if bool(Rec_dictionary) == False:
            Rec_dictionary['States'] = S0
            Rec_dictionary['V_Estimate'] = V_S0
            Rec_dictionary['Advantage'] = GAE
            Rec_dictionary['Actions'] = Actions
            Rec_dictionary['Sig'] = Sigmas
        else:
            Rec_dictionary['States'] = np.concatenate((Rec_dictionary['States'], S0), axis = 0)
            Rec_dictionary['V_Estimate'] = np.concatenate((Rec_dictionary['V_Estimate'], V_S0), axis = 0)
            Rec_dictionary['Advantage'] = np.concatenate((Rec_dictionary['Advantage'], GAE), axis = 0)
            Rec_dictionary['Actions'] = np.concatenate((Rec_dictionary['Actions'], Actions), axis = 0)
            Rec_dictionary['Sig'] = np.concatenate((Rec_dictionary['Sig'], Sigmas), axis = 0)

        return Rec_dictionary
    
    
    def Fit_Model(self, Epochs, Exp):
        n = Exp['States'].shape[0]
        ind = np.random.choice(n, size=(Epochs, n//Epochs), replace=False)
        
        for k in range(Epochs):
            self.TF_Session.run(self.AC_Net.fit, feed_dict = {self.AC_Net.X:Exp['States'][ind[k],:], self.AC_Net.A_in:Exp['Actions'][ind[k],:], self.AC_Net.V_in:Exp['V_Estimate'][ind[k],:],
                            self.AC_Net.sigma_in: Exp['Sig'][ind[k],:], self.AC_Net.Adv:Exp['Advantage'][ind[k],:], self.AC_Net.learning_rate:self.Learning_Rate})    
    
    def Decide_Move(self, state, Sigma):
        Mu = self.TF_Session.run(self.AC_Net.Predict, feed_dict = {self.AC_Net.X: state.reshape(-1, self.State_Dim)})
        return np.random.normal(Mu, Sigma)
    
    
   # Auxillary Functions
    def Print_V_Function (self):
        V_Function = []

        dx = (self.Environment.observation_space.high[0] - self.Environment.observation_space.low[0]) / 50
        dy = (self.Environment.observation_space.high[1] - self.Environment.observation_space.low[1]) / 50

        x_low = self.Environment.observation_space.low[0]
        y_low = self.Environment.observation_space.low[1]

        for i in range(50):
            for j in range(50):
                State = np.array([x_low + dx * i, y_low + dy * j]).reshape((-1,self.State_Dim))
                Value = self.TF_Session.run(self.AC_Net.Value_Pred, feed_dict = {self.AC_Net.X: State})
                V_Function.append([x_low + i * dx, y_low + j * dy, Value[0][0]])
        V_Function = np.array(V_Function)
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(V_Function[:,0], V_Function[:,1], V_Function[:,2], zdir='z', c= 'red')
        plt.show()    
    
    
    def Print_Action_Space (self):
         A_Function = []
    
         dx = (self.Environment.observation_space.high[0] - self.Environment.observation_space.low[0]) / 50
         dy = (self.Environment.observation_space.high[1] - self.Environment.observation_space.low[1]) / 50
    
         x_low = self.Environment.observation_space.low[0]
         y_low = self.Environment.observation_space.low[1]
    
         for i in range(50):
             for j in range(50):
                 State = np.array([x_low + dx * i, y_low + dy * j]).reshape(-1,self.State_Dim)
                 Action = self.TF_Session.run(self.AC_Net.Predict, feed_dict = {self.AC_Net.X: State})
                 A_Function.append([x_low + i * dx, y_low + j * dy, Action[0][0]])
                 
         A_Function = np.array(A_Function)
         fig = plt.figure()
         ax = fig.add_subplot(111, projection='3d')
         ax.scatter(A_Function[:,0], A_Function[:,1], A_Function[:,2], zdir='z', c= 'red')
         plt.show()
    
    
        
    
    
    
    
    

    
    
    
    

    
    