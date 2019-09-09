# -*- coding: utf-8 -*-
"""
Created on Fri Jul 26 08:44:20 2019

@author: hydra li
"""
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import copy
import math
import random
import pickle
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)

# In this version of Value Actor_Critic we got rid of the concept of Q function and instead just used one neural network to output state value and policy
class Pnetwork():
    def __init__(self, Shape, S_dim, A_dim, Learning_Rate, Name, between_layer = 'Sigmoid'):
        #Same placeholder and initialiser as the Policy Actor Critic
        self.States  = tf.placeholder(shape = [None,S_dim], dtype = tf.float32, name = 'States')
        self.Actions  = tf.placeholder(shape = [None,A_dim], dtype = tf.float32, name = 'Actions')      
        self.initializer   = tf.contrib.layers.xavier_initializer()
        self.learning_rate = Learning_Rate

        if between_layer =='Sigmoid':
            Activation = tf.nn.sigmoid
        elif between_layer =='Relu':
            Activation = tf.nn.relu
            
        #From this line to line 56 we layout the Structure of the Network. 
        #State value and Policy parameters have shared previous layer and differ in last layer's shape and activation function.
        Shape = [S_dim] + Shape 
        Weights = []
        Layers  = []
        for i in range(len(Shape) - 1):
            Name_ = Name + str(i)
            i_    = Shape[i]
            j_    = Shape[i+1]
            Weights.append(tf.get_variable(Name_, shape = [i_, j_], initializer = self.initializer))
        Weights.append(tf.get_variable("Value_Weights", shape = [Shape[-1], 1], initializer = self.initializer))
        Weights.append(tf.get_variable("Mu_Weights", shape = [Shape[-1], A_dim], initializer = self.initializer))
        Weights.append(tf.get_variable("Sigma_Weights", shape = [Shape[-1], A_dim], initializer = self.initializer))
                
        for j in range(len(Weights) - 3):
            if j == 0:
                Layers.append(Activation(tf.matmul(self.States,Weights[j])))
            else:
                Layers.append(Activation(tf.matmul(Layers[j-1], Weights[j])))
        
        self.Values = tf.matmul(Layers[-1], Weights[-3])        
        self.Mus     =  tf.nn.tanh(tf.matmul(Layers[-1], Weights[-2]))
        Sigmas_  =  tf.nn.softplus(tf.matmul(Layers[-1], Weights[-1]))
        self.Sigmas  =  tf.clip_by_value(Sigmas_, 0.05, 1)


        #Set a place holder for the Target V calculated from Value estimate from network and Bellman Equation. Total loss is made of three parts, policy, value and Entropy Bonus. Infer the Advantage from Target_V
        #then calculate the policy loss and Entropy Bonus based on Mu, Sigma predicted from network. Value loss is calculated using l2 loss and use accumulative rewards as target V
        self.Target_V = tf.placeholder(shape = [None, 1], dtype = tf.float32, name = 'Target_V')
        Advantage = self.Target_V - self.Values
        
        LogLik_    = - (tf.log(tf.sqrt(2 * math.pi * tf.square(self.Sigmas)))/2 + tf.square(self.Actions - self.Mus) / (2 * tf.square(self.Sigmas)))
        self.LogLik    = tf.reduce_sum(LogLik_, axis = 1)
        self.Policy_loss  =  -tf.matmul(tf.reshape(self.LogLik, shape = [1,-1]), Advantage)
        
        self.Entropy_Decay = tf.placeholder(tf.float32, shape = ())        
        self.Entropy   = tf.reduce_sum(tf.log(self.Sigmas) + 1/2 + np.log(math.pi*2)/2)*self.Entropy_Decay
        
        self.Value_loss = tf.reduce_sum(tf.square(Advantage))
        self.Loss      = self.Policy_loss + 0.5*self.Value_loss - self.Entropy
        
        # Then we define two different fittings where we can fit the entire model, or fit towards policy loss only 
        #This design is for the purpose of allowing different fitting frequency. 
        self.trainer   = tf.train.AdamOptimizer(learning_rate = self.learning_rate)
        self.fit       = self.trainer.minimize(self.Loss)
        self.Policy_fit = self.trainer.minimize(self.Policy_loss - self.Entropy)



#Experience Pool is a class for the purpose of storing experience and draw samples
class Experience_Pool():
    #In order to store the experience in 2d np.array to accelerate we need to organise the experiences into steps of [State, Action, Reward, Transition, Down, Cumulative_Reward] and only store values
    #Current Pool upperlimit set to be 3 million steps, you should adjust according to your computer RAM size.
    def __init__(self, State_dim, Action_dim, max_length = 3000000):
        self.total_dim = State_dim + Action_dim + 1 + State_dim + 1 + 1
        self.Pool = np.zeros((1, self.total_dim))
        self.max_length = max_length
        self.State_dim = State_dim
        self.Action_dim = Action_dim
        
    #Add an entire episode to pool when an episode is completed
    def add(self, experience, discount = 1):
        experience = np.array(experience).reshape(-1, self.total_dim)
        
        #Here I calculate cumulative rewards
        for K in range(experience.shape[0] - 1)[::-1]:
            experience[K, -1] = experience[K+1 , -1]*discount + experience[K, -1]
            
        n = max(self.Pool.shape[0] + experience.shape[0] - self.max_length, 0)
        self.Pool = np.vstack((self.Pool[n:, :], experience))
    
    #This function samples corresponding Records from the pool. Reason why it's so messy with many choices is to accelerate computation by elimating all non_essential pass_by_values
    def sample(self, N = 'All', state = False, action = False, reward = False, Transition = False, Q = False, done = False):
        states = None
        actions = None
        rewards = None
        Transitions = None
        Q_Value = None
        Done = None
        if N == 'All':
            if state == True:
                states = self.Pool[:, :self.State_dim]
            if action == True:
                actions = self.Pool[:, self.State_dim:(self.State_dim + self.Action_dim)]
            if reward == True:
                rewards = self.Pool[:, (self.State_dim + self.Action_dim)]
                rewards = rewards.reshape(-1,1)
            if Transition == True:
                Transitions = self.Pool[:, (self.State_dim + self.Action_dim + 1):(self.State_dim + self.Action_dim + self.State_dim + 1)]
            if done == True:
                Done = self.Pool[:, -2]
                Done = Done.reshape(-1,1)
            if Q == True:
                Q_Value = self.Pool[:, -1]
                Q_Value = Q_Value.reshape(-1,1)

        if N != 'All':
            idx = np.random.choice(self.Pool.shape[0], N)
            if state == True:
                states = self.Pool[idx, :self.State_dim]
            if action == True:
                actions = self.Pool[idx, self.State_dim:(self.State_dim + self.Action_dim)]
            if reward == True:
                rewards = self.Pool[idx, (self.State_dim + self.Action_dim)]
                rewards = rewards.reshape(-1,1)
            if Transition == True:
                Transitions = self.Pool[idx, (self.State_dim + self.Action_dim):(self.State_dim + self.Action_dim + self.State_dim)]
            if done == True:
                Done = self.Pool[idx, -2]
                Done = Done.reshape(-1,1)
            if Q == True:
                Q_Value = self.Pool[idx, -1]
                Q_Value = Q_Value.reshape(-1,1)
        
        return states, actions, rewards, Transitions, Q_Value, Done

#---------------------------------------------------------------------------------------------------------------------------------------------------------
#This is the Main class. 
class A2C_Agent_Value():
    # Essential parameters to construct an agent includes:
    # 1. It needs to know the environemtn to work on. 2. It needs to know the network shape 3. Entropy to encourage exploration, this may also be decreased over episodes.
    # 4. Over_Sample is a choice to allow for separating two experience pool or one combined pool, where Positive_Level is used to put experience in correct pool. Min_Exp and Min_Pos_Exp before first fitting.
    # 6. Update_Offline_Value is the frequency to update the entire network based on total loss, if Update_Policy is otherwise specified, policy can be updated more frequency with smaller epochs
    # 7. Other parameters are standard, including learning rate, replay ratio decides sample size. Jump start allows to skip initialisation by loading pre-stored success experiences 
    
    def __init__ (self, Env, Network_Size, Entropy, Min_Exp = 10000, Min_Pos_Exp = 10000, Gamma = 0.999, Replay_Ratio = 2, Update_Offline_Value = 25, 
                  Update_Policy = None, Policy_Epoch = 1, Value_Epoch = 50, Learning_Rate_P = 0.001, Over_Sample = False, Positive_Level = 0.0,  Jump_start = False):

        self.Env = Env
        self.State_dim = Env.observation_space.low.size
        self.Action_dim = Env.action_space.low.size
        
        self.Policy_Model = Pnetwork(Network_Size, self.State_dim, self.Action_dim, Learning_Rate_P, "Policy_Network")
        self.TF_Session   = tf.Session()                                       #Store TF session across entire training period.
        self.TF_Session.run(tf.global_variables_initializer())                 #Initialise all TF variables before they can be used

        self.Exp = Experience_Pool(self.State_dim, self.Action_dim)            #Initilise the experience pool
        self.Pos_Exp = Experience_Pool(self.State_dim, self.Action_dim)     

        self.Positive_Level = Positive_Level
        self.Over_Sample    = Over_Sample                                      # If Over_Sample == True, then  experiance replay will be 50% Pos_Exp

        self.Min_Exp     = Min_Exp
        self.Min_Pos_Exp = Min_Pos_Exp

        self.Update_Value = Update_Offline_Value
        self.Update_Policy = Update_Policy

        self.Gamma = Gamma
        self.Replay_Ratio = Replay_Ratio
        self.Policy_Epoch = Policy_Epoch
        self.Epoch = Value_Epoch
        
        self.Jump_start = Jump_start
        self.Entropy  =  Entropy

    def Initialise_Experiance (self):
        # Gather some expereience by acting randomly and store it, or load pre_stored experiences
        if self.Jump_start:
            with open('Initiation_pos.txt', 'rb') as fp:
                self.Pos_Exp = pickle.load(fp)
        
        i = 0    #i keep on track how the experience is going and report every 100 episodes. First Initialisation can be lengthy
        while self.Exp.Pool.shape[0] < self.Min_Exp or (self.Pos_Exp.Pool.shape[0] < self.Min_Pos_Exp and self.Over_Sample == True) :
            State_0 = self.Env.reset()
            Done = False
            Episode_Exp = []
            i += 1
            if i % 100 == 0:
                print("Episode " + str(i) + "    Exp Pool : " + str(self.Exp.Pool.shape[0]) + "    Pos_Exp Pool : " + str(self.Pos_Exp.Pool.shape[0]))
             
            #Get Episode experience   
            while Done == False:
                Action = self.Env.action_space.sample()
                State_1, Reward, Done, Info = self.Env.step(Action)
                This_step = list(State_0) + list(Action) + [Reward] + list(State_1) + [Done] + [Reward]
                Episode_Exp += This_step
                State_0 = State_1
                
                #Add episode experience to correct pool
                if Done == True:
                    if Reward > self.Positive_Level and self.Pos_Exp.Pool.shape[0] < self.Min_Pos_Exp and self.Over_Sample == True:
                        self.Pos_Exp.add(Episode_Exp, self.Gamma)
                        
                    elif self.Exp.Pool.shape[0] < self.Min_Exp:
                        self.Exp.add(Episode_Exp, self.Gamma)

        # To do the first fitting we used the entire experience pool Our first Advantage estimate before Q network exist is just accumulative reward minus its mean
        States, Actions, _, _, Q_values, _ = self.Pos_Exp.sample(state= True, action= True, Q= True)   
        
        if self.Over_Sample == True:
            States1, Actions1,_, _, Q_values1, _ = self.Exp.sample(state= True, action= True, Q= True)          
            States = np.vstack((States,States1))
            Actions = np.vstack((Actions, Actions1))
            Q_values = np.vstack((Q_values, Q_values1))
        
        # In the first fitting we use 250 iterations/Epochs       
        for i in range(250):
            self.TF_Session.run(self.Policy_Model.fit, feed_dict = {self.Policy_Model.States:States,
                                                                    self.Policy_Model.Actions: Actions,
                                                                    self.Policy_Model.Target_V: Q_values,
                                                                    self.Policy_Model.Entropy_Decay: self.Entropy})

    #Training function that trains both networks over N_episodes
    def Train (self, N_Episodes):
        # Check if the expereience pool has been initialised before training
        if self.Exp.Pool.shape[0] < self.Min_Exp:
            self.Initialise_Experiance()
        #Calculate the decaying speed such that by the end of N_Episode the Entropy bonus decays to 1/10 of its original
        self.Entropy_Decay = 0.1**(self.Update_Value/N_Episodes)
        
        ##Store some key Metrics to understand how the agent is learning
        Episode_Rewards   = []
        Episode_Sigma     = []
        Fit_Entropy       = []
        Fit_Loss          = []
        
        for i in range(N_Episodes):
            State_0 = self.Env.reset()
            Done = False
            # Check whether we need to refit Whole Network and then refit
            if i % self.Update_Value == 0:
                Ent, Loss = self.Update_Networks()
                Fit_Entropy.append(Ent)
                Fit_Loss.append(Loss[0][0])
                
            # Check if we need to refit Policy and refit. In this model I think it is preferrable not to update Value and policy at different frequency so I commented following codes out
#            if self.Update_Policy != None:
#                if i % self.Update_Policy == 0:
#                    self.Update_Networks(only_policy = True)

            ##Collect Episode experience to add to pool and collect performance metrics
            Episode_Experiance = []
            local_sigma = 0
            local_reward = 0
            while Done == False:
                Action, Sigma = self.Decide_Action(State_0.reshape(1, -1))
                Action = Action[0]
                State_1, Reward, Done, Info = self.Env.step(Action)
                local_sigma += Sigma[0]
                local_reward += Reward
                This_step = list(State_0) + list(Action) + [Reward] + list(State_1) + [Done] + [Reward]
                Episode_Experiance += This_step
                State_0 = State_1
            Episode_Rewards.append(local_reward)
            Episode_Sigma.append(local_sigma/len(Episode_Experiance)*(self.Action_dim + 2*self.State_dim + 3))
 
            # By the end of Episode decide which pool this Episode should go to
            if local_reward > self.Positive_Level and self.Over_Sample == True:
                self.Pos_Exp.add(Episode_Experiance, self.Gamma)
            else: 
                self.Exp.add(Episode_Experiance, self.Gamma)
                
            #Keep track of training progress
            if i % 10 == 0:
                print(str(i), end = " ")
                   
        return Episode_Rewards, Episode_Sigma, Fit_Entropy, Fit_Loss

    #Refit the Models
    def Update_Networks (self, only_policy = False):
        #Draws sample from Experience pool and use them to refit the model.
        Target, Actions, States = self.Sample_Exp(only_policy)
        
        #Again It is possible to only update towards policy loss but it's not called in the experiment
        if only_policy:
            for k in range(self.Policy_Epoch):
                self.TF_Session.run(self.Policy_Model.Policy_fit, feed_dict = {self.Policy_Model.States:States,
                                                                                self.Policy_Model.Actions : Actions,
                                                                                self.Policy_Model.Target_V : Target,
                                                                                self.Policy_Model.Entropy_Decay : self.Entropy})
        #To fit the model and collect key metrcs about how it's learning    
        else:
#            self.Entropy*=self.Entropy_Decay
            for k in range(self.Epoch):
                self.TF_Session.run(self.Policy_Model.fit, feed_dict = {self.Policy_Model.States:States,
                                                                        self.Policy_Model.Actions : Actions,
                                                                        self.Policy_Model.Target_V : Target,
                                                                        self.Policy_Model.Entropy_Decay : self.Entropy})
    
            Ent = self.TF_Session.run(self.Policy_Model.Entropy, feed_dict = {self.Policy_Model.States:States,
                                                                    self.Policy_Model.Actions : Actions,
                                                                    self.Policy_Model.Target_V : Target,
                                                                    self.Policy_Model.Entropy_Decay : self.Entropy})
    
            Los = self.TF_Session.run(self.Policy_Model.Loss, feed_dict = {self.Policy_Model.States:States,
                                                                    self.Policy_Model.Actions : Actions,
                                                                    self.Policy_Model.Target_V : Target,
                                                                    self.Policy_Model.Entropy_Decay : self.Entropy})
            return Ent, Los
    

    
    def Sample_Exp (self, only_policy = False):
        #Check whether or not we need to oversample, if so sample half half from each pool and combine them, otherwise simply sample from main pool
        if self.Over_Sample == True:
            N = int(min(self.Update_Value * 800 * self.Replay_Ratio / 2, self.Pos_Exp.Pool.shape[0], self.Exp.Pool.shape[0]))
            #Use a smaller batch size if we are just refitting the policy
            if only_policy:
                N = int(N/5)
            #Sample two pools and combine
            States, Actions, Reward, Future_States, _, Terminal_Step = self.Pos_Exp.sample(N, state= True, action= True, reward= True, Transition= True, done= True)
            States1, Actions1, Reward1, Future_States1, _, Terminal_Step1 = self.Exp.sample(N, state= True, action= True, reward= True, Transition= True, done= True)
            States = np.vstack((States,States1))
            Actions = np.vstack((Actions, Actions1))
            Future_States = np.vstack((Future_States, Future_States1))
            Terminal_Step = np.vstack((Terminal_Step, Terminal_Step1))
            Reward = np.vstack((Reward, Reward1))
            
        else:
            #Sample One pool
            N = int(min(self.Update_Value * 800 * self.Replay_Ratio, self.Exp.Pool.shape[0]))
            if only_policy:
                N = int(N/10)
            States, Actions, Reward, Future_States, _, Terminal_Step = self.Exp.sample(N, state= True, action= True, reward= True, Transition= True, done= True)
        
        #Output the State, Action, and Advantage that we need in fitting model
        Future_Values  = self.TF_Session.run(self.Policy_Model.Values, feed_dict = {self.Policy_Model.States : Future_States})
        Target = Reward + Future_Values*(1-Terminal_Step.astype(int))* self.Gamma 
        
        return Target, Actions, States

    #Function to decide action given state
    def Decide_Action (self, State, optimal = False):
        Mu    = self.TF_Session.run(self.Policy_Model.Mus, feed_dict = {self.Policy_Model.States:State})
        if optimal == True:
            Action = np.clip(Mu, self.Env.action_space.low, self.Env.action_space.high)
            return Action.reshape(-1,self.Action_dim)
        
        Sigma = self.TF_Session.run(self.Policy_Model.Sigmas, feed_dict = {self.Policy_Model.States:State})                
        Action = np.clip(np.random.normal(Mu, Sigma), self.Env.action_space.low, self.Env.action_space.high)
        return Action.reshape(-1,self.Action_dim), Sigma


# Auxillary Functions ---------------------------------------------------------------------------------------------------------------
    def Print_V_Function (self):
         V_Function = []
    
         dx = (self.Env.observation_space.high[0] - self.Env.observation_space.low[0]) / 50
         dy = (self.Env.observation_space.high[1] - self.Env.observation_space.low[1]) / 50
    
         x_low = self.Env.observation_space.low[0]
         y_low = self.Env.observation_space.low[1]
    
         for i in range(50):
             for j in range(50):
                 State = np.array([x_low + dx * i, y_low + dy * j]).reshape(-1,self.State_dim)
                 Value = self.TF_Session.run(self.Policy_Model.Values, feed_dict = {self.Policy_Model.States : State})
                 V_Function.append([x_low + i * dx, y_low + j * dy, Value[0][0]])
         V_Function = np.array(V_Function)
         
         fig = plt.figure()
         ax = fig.add_subplot(111, projection='3d')
         ax.scatter(V_Function[:,0], V_Function[:,1], V_Function[:,2], zdir='z', c= 'red')
         plt.show()
    

    def Print_Action_Space (self):
         A_Function = []
    
         dx = (self.Env.observation_space.high[0] - self.Env.observation_space.low[0]) / 50
         dy = (self.Env.observation_space.high[1] - self.Env.observation_space.low[1]) / 50
    
         x_low = self.Env.observation_space.low[0]
         y_low = self.Env.observation_space.low[1]
    
         for i in range(50):
             for j in range(50):
                 State = np.array([x_low + dx * i, y_low + dy * j]).reshape(-1,self.State_dim)
                 Action = self.Decide_Action(State, optimal= True)
                 A_Function.append([x_low + i * dx, y_low + j * dy, Action[0][0]])
                 
         A_Function = np.array(A_Function)
         fig = plt.figure()
         ax = fig.add_subplot(111, projection='3d')
         ax.scatter(A_Function[:,0], A_Function[:,1], A_Function[:,2], zdir='z', c= 'red')
         plt.show()
         
         
    def Print_Sigma_Space (self):
         S_Function = []
    
         dx = (self.Env.observation_space.high[0] - self.Env.observation_space.low[0]) / 50
         dy = (self.Env.observation_space.high[1] - self.Env.observation_space.low[1]) / 50
    
         x_low = self.Env.observation_space.low[0]
         y_low = self.Env.observation_space.low[1]
    
         for i in range(50):
             for j in range(50):
                 State = [x_low + dx * i, y_low + dy * j]
                 State = np.array(State).reshape(1,-1)
                 Value = self.TF_Session.run(self.Policy_Model.Sigmas, feed_dict = {self.Policy_Model.States :State})
                 S_Function.append([x_low + i * dx, y_low + j * dy, Value[0][0]])
         S_Function = np.array(S_Function)
         
         fig = plt.figure()
         ax = fig.add_subplot(111, projection='3d')
         ax.scatter(S_Function[:,0], S_Function[:,1], S_Function[:,2], zdir='z', c= 'red')
         plt.show()
         

    def Record_Experience(self):
        with open('Initiation_pos.txt', 'wb') as fp:
            pickle.dump(self.Pos_Exp, fp)

                









