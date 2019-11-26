import numpy             as np
import pandas            as pd
import multiprocessing   as mp
import matplotlib.pyplot as plt
import copy
import gym
import Portfolio_Gym
import itertools
import A2C_X as A2C


# Create the list of hypers to loop through.

class HPC_Wrapper:

    '''
    Allows for the following performance metrics:
    1. Agent Repeatability - How much variation does re-instancing the agent create
    2. Grid Search Hyper parameter tuning
    3. Genetic Hyper parameter tuning


    How to use
    ----------
        1. Instance the class, sending the Env_String for gym.make.
        2. (Optional) If the Env should be created with custom settigns, pass these arguments to HPC_Wrapper.Set_Env_Params()
        3. Call one of the three functions:
            1. Hyper_Grid_Search
            2. Agent_Repeatability
            3. Hyper_Genetic_Search

    '''

    def __init__ (self, Env_String, N_Eps = 100000, N_Instances = 64, Over_Sample_Mult = 0):

        self.Env_String = Env_String
        self.N_Eps = N_Eps
        self.Env_Params = None
        self.N_Instances = N_Instances
        self.Over_Sample_Mult = Over_Sample_Mult


    def Set_Env_Params (self, **kwargs):
        self.Env_Params = kwargs


    def Hyper_Grid_Search (self, **kwargs):

        Actor_Learning_Rates  = kwargs['Actor_Learning_Rate'] if 'Actor_Learning_Rate' in kwargs.keys() else [0.005]
        Actor_Epochs          = kwargs['Actor_Epoch']         if 'Actor_Epoch'         in kwargs.keys() else [1]
        Actor_Sizes           = kwargs['Actor_Network_Size']  if 'Actor_Network_Size'  in kwargs.keys() else [[8,4]]
        Actor_Batch_Sizes     = kwargs['Actor_Batch_Size']    if 'Actor_Batch_Size'    in kwargs.keys() else [60]
        Actor_Alphas          = kwargs['Actor_Alpha']         if 'Actor_Alpha'         in kwargs.keys() else [0.1]

        Critic_Learning_Rates = kwargs['Critic_Learning_Rate'] if 'Critic_Learning_Rate' in kwargs.keys() else [0.005]
        Critic_Epochs         = kwargs['Critic_Epoch']         if 'Critic_Epoch'         in kwargs.keys() else [10]
        Critic_Sizes          = kwargs['Critic_Network_Size']  if 'Critic_Network_Size'  in kwargs.keys() else [[8,4]]
        Critic_Batch_Sizes    = kwargs['Critic_Batch_Size']    if 'Critic_Batch_Size'    in kwargs.keys() else [60]
        Critic_Alphas         = kwargs['Critic_Alpha']         if 'Critic_Alpha'         in kwargs.keys() else [0.1]

        Gamma             = kwargs['Gamma']             if 'Gamma'             in kwargs.keys() else [0.999]
        Sigma_Range       = kwargs['Sigma_Range']       if 'Sigma_Range'       in kwargs.keys() else [[2, 0.5]]
        Sigma_Anneal      = kwargs['Sigma_Anneal']      if 'Sigma_Anneal'      in kwargs.keys() else [1]
        Retrain_Frequency = kwargs['Retrain_Frequency'] if 'Retrain_Frequency' in kwargs.keys() else [20]
        Action_Space_Clip = kwargs['Action_Space_Clip'] if 'Action_Space_Clip' in kwargs.keys() else [75]
        Experiance_Mode   = kwargs['Experiance_Mode']   if 'Experiance_Mode'   in kwargs.keys() else ['TD_Lambda']
        TD_Lambda         = kwargs['TD_Lambda']         if 'TD_Lambda'         in kwargs.keys() else [0.50]
        Monte_Carlo_Frac  = kwargs['Monte_Carlo_Frac']  if 'Monte_Carlo_Frac'  in kwargs.keys() else [0.2]

        Combinations = [i for i in itertools.product(Actor_Learning_Rates, Actor_Epochs, Actor_Sizes, Actor_Batch_Sizes, Actor_Alphas,
                                                     Critic_Learning_Rates, Critic_Epochs, Critic_Sizes, Critic_Batch_Sizes, Critic_Alphas,
                                                     Gamma, Sigma_Range, Sigma_Anneal, Retrain_Frequency, Action_Space_Clip, Experiance_Mode, TD_Lambda, Monte_Carlo_Frac)]

        Hypers = []
        for i, Comb in enumerate(Combinations):

            Hypers.append({'Actor_Learning_Rate'  : Comb[0],
                           'Actor_Epoch'          : Comb[1],
                           'Actor_Network_Size'   : Comb[2],
                           'Actor_Activation'     : 'Sigmoid',
                           'Actor_Batch_Size'     : Comb[3],
                           'Actor_Alpha'          : Comb[4],

                           'Critic_Learning_Rate' : Comb[5],
                           'Critic_Epoch'         : Comb[6],
                           'Critic_Network_Size'  : Comb[7],
                           'Critic_Activation'    : 'Sigmoid',
                           'Critic_Batch_Size'    : Comb[8],
                           'Critic_Alpha'         : Comb[9],

                           'Gamma'             : Comb[10],
                           'Sigma_Range'       : Comb[11],
                           'Sigma_Anneal'      : Comb[12],
                           'Retrain_Frequency' : Comb[13],
                           'Action_Space_Clip' : Comb[14],
                           'Experiance_Mode'   : Comb[15],
                           'TD_Lambda'         : Comb[16],
                           'Monte_Carlo_Frac'  : Comb[17]})

        Hypers_DataFrame = pd.DataFrame(Hypers)
        Results_DataFrame = pd.DataFrame()

        for i, hyper in enumerate(Hypers):
            # Run the investigation...
            with mp.Pool(mp.cpu_count()) as pool:
                Input = [{'AC_Hypers' : hyper, 'Seed' : np.random.randint(0, 1e9)} for i in range(64)]
                results = np.array(pool.map(self.Run, Input))

            # Now format the results.
            # For each hyper set we want the Sharpe Ratio and Delta Utility of each instance.
            Results_DataFrame['Sharpes_' + str(i)] = results[:,0]
            Results_DataFrame['Delta_Utility_' + str(i)] = results[:,1]

        Hypers_DataFrame.to_csv('Hypers_Key.csv')
        Results_DataFrame.to_csv('Results.csv')


    def Agent_Repeatability (self, **kwargs):


        Actor_Hypers  = {"Learning Rate" : kwargs['Actor_Learning_Rate']  if 'Actor_Learning_Rate' in kwargs.keys() else 0.005,
                         "Epoch"         : kwargs['Actor_Epoch']          if 'Actor_Epoch'         in kwargs.keys() else 1,
                         "Network Size"  : kwargs['Actor_Size']           if 'Actor_Size'          in kwargs.keys() else [8,4],
                         "Activation"    : "Sigmoid",
                         "Batch Size"    : kwargs['Actor_Batch_Size']     if 'Actor_Batch_Size'    in kwargs.keys() else 60,
                         "Alpha"         : kwargs['Actor_Alpha']          if 'Actor_Alpha'         in kwargs.keys() else 0.1}

        Critic_Hypers = {"Learning Rate" : kwargs['Critic_Learning_Rate'] if 'Critic_Learning_Rate' in kwargs.keys() else 0.005,
                         "Epoch"         : kwargs['Critic_Epoch']         if 'Critic_Epoch'         in kwargs.keys() else 10,
                         "Network Size"  : kwargs['Critic_Size']          if 'Critic_Size'          in kwargs.keys() else [8,4],
                         "Activation"    : "Sigmoid",
                         "Batch Size"    : kwargs['Critic_Batch_Size']    if 'Critic_Batch_Size'    in kwargs.keys() else 60,
                         "Alpha"         : kwargs['Critic_Alpha']         if 'Critic_Alpha'         in kwargs.keys() else 0.1}

        Hypers = {'Actor_Hypers'      : Actor_Hypers,
                  'Critic_Hypers'     : Critic_Hypers,
                  'Gamma'             : kwargs['Gamma']             if 'Gamma'             in kwargs.keys() else 0.999,
                  'Sigma_Range'       : kwargs['Sigma_Range']       if 'Sigma_Range'       in kwargs.keys() else [2, 0.5],
                  'Sigma_Anneal'      : kwargs['Sigma_Anneal']      if 'Sigma_Anneal'      in kwargs.keys() else 1,
                  'Retrain_Frequency' : kwargs['Retrain_Frequency'] if 'Retrain_Frequency' in kwargs.keys() else 20,
                  'Action_Space_Clip' : kwargs['Action_Space_Clip'] if 'Action_Space_Clip' in kwargs.keys() else 75,
                  'Experiance_Mode'   : kwargs['Experiance_Mode']   if 'Experiance_Mode'   in kwargs.keys() else 'TD_Lambda',
                  'TD_Lambda'         : kwargs['TD_Lambda']         if 'TD_Lambda'         in kwargs.keys() else 0.50,
                  'Monte_Carlo_Frac'  : kwargs['Monte_Carlo_Frac']  if 'Monte_Carlo_Frac'  in kwargs.keys() else 0.2}

        Env = gym.make(self.Env_String)
        Env.Set_Params(**self.Env_Params)
        if self.Over_Sample_Mult != 0 : Env.Over_Sample(Mult = self.Over_Sample_Mult, N_ = 5)

        with mp.Pool(mp.cpu_count()) as pool:
            Input = [{'AC_Hypers' : Hypers, 'Seed' : np.random.randint(0, 1e9), 'Env' : Env} for _ in range(self.N_Instances)]
            results = np.array(pool.map(self.Run, Input))

        return results


    def Hyper_Genetic_Search (self, Population = 64, Mutate_Prob = 0.1, Num_Generations = 20):

        '''
        Parameters
        ----------
            Population | int
                The number of instances which make up a single generation.

            Mutate_Prob | float
                The probability that a child will develop a mutation.

            Num_Generations | int
                The number of generations before the algorithm stops.

        Notes
        -----
            The performance of the instances is measured in terms of their Delta Utility.

            The Genetic algorithm follows the following steps:
                1. Generate N instances of the AC by drawing their parameters from a uniform distribution between specified bounds
                2. Train each instance and record a performance metric (here using delta utility)
                3. Use the current generation of instances to create a new generation (half the population will be replaced with the new generation). Child instances are created as follows:
                    1. Select 10 instances at random
                    2. Order them based upon their performance
                    3. Select two, weighting each instance's likelihood of being picked by e^-i/5
                    4. For each hyperparameter pick randomly from the two parents.
                    5. With a probaility of Mutate_Prob, redraw one of the parameters from its original distribution
                4. Return to step 2 and repeat.
        '''

        def Gen_Initial_Parameters ():
            '''
            Genrates a set of hyperparameters for A2C_X, within safe (hardcoded) boundaries.
            '''

            Hyper = {'Actor_Learning_Rate'  : np.random.uniform(0.0001, 0.01),
                     'Actor_Epoch'          : np.random.randint(1, 10),
                     'Actor_Network_Size'   : np.random.randint(2, 16, size = np.random.randint(1, 5)),
                     'Actor_Activation'     : 'Sigmoid',
                     'Actor_Batch_Size'     : np.random.randint(30, 240),
                     'Actor_Alpha'          : np.random.uniform(0.001, 0.5),

                     'Critic_Learning_Rate' : np.random.uniform(0.0001, 0.01),
                     'Critic_Epoch'         : np.random.randint(1, 10),
                     'Critic_Network_Size'  : np.random.randint(2, 16, size = np.random.randint(1, 5)),
                     'Critic_Activation'    : 'Sigmoid',
                     'Critic_Batch_Size'    : np.random.randint(30, 240),
                     'Critic_Alpha'         : np.random.uniform(0.001, 0.5),

                     'Gamma'             : np.random.uniform(0.2, 1),
                     'Sigma_Range'       : [np.random.uniform(1.5, 3), np.random.uniform(0.25, 1)],
                     'Sigma_Anneal'      : 1,
                     'Retrain_Frequency' : np.random.randint(10, 100),
                     'Action_Space_Clip' : 75,
                     'Experiance_Mode'   : 'TD_Lambda',
                     'TD_Lambda'         : np.random.uniform(0, 1),
                     'Monte_Carlo_Frac'  : np.random.uniform(0, 0.2)}

            return Hyper


        def Merge_And_Mutuate (Hyper_A, Hyper_B):
            '''
            Merges two hyperparameter dictionaries
            '''

            Hyper_C = dict()

            for key in Hyper_A.keys():
                Hyper_C[key] = Hyper_A[key] if np.random.uniform() > 0.5 else Hyper_B[key]

            if np.random.uniform() < Mutate_Prob:
                Mutant_Hypers = Gen_Initial_Parameters()
                key = np.random.choice(list(Hyper_C.keys()))
                Hyper_C[key] = Mutant_Hypers[key]

            return Hyper_C


        Prob_Mask = np.array([np.exp(-i / 5) for i in range(10)])
        Prob_Mask = Prob_Mask / np.sum(Prob_Mask)

        Hypers = []

        for _ in range(Num_Generations):

            if len(Hypers) == 0:
                for i in range(Population):
                    Hypers.append(Gen_Initial_Parameters())

            else:
                New_Hypers = []
                # Breed some children
                for i in range(int(Population / 2)):

                    B = np.random.choice(np.arange(Population), 10, replace = False)
                    C = np.argsort(results[B])
                    D = np.random.choice(np.arange(10), 2, p = Prob_Mask, replace = False)

                    New_Hypers.append(Merge_And_Mutuate(Hypers[B[C[D]][0]], Hypers[B[C[D]][1]]))

                # Now kill the weak hypers
                Hypers = list(np.array(Hypers)[np.argsort(results)][0:int(Population/2)])
                Hypers.extend(New_Hypers)


            Input = []
            for i in range(len(Hypers)):
                Input.append({'Seed' : np.random.randint(0, 1e9), 'AC_Hypers' : Hypers[i]})

            with mp.Pool(mp.cpu_count()) as pool:
                results = np.array(pool.map(self.Run, Input))[:,1]


        # Finally print the final generation and their scores to a csv
        for i in range(len(Hypers)):
            Hypers[i]['Score'] = results[i]

        Data = pd.DataFrame(Hypers)
        Data.to_csv('Genetic_Tuning_Output.csv')


    def Run (self, Input):

        '''
        Function to facilitate multiprocessing.

        Paramters
        ---------
            Input | dict
                A dictionary with the following keys:
                    Seed | int - The seed for numpy random number generator to ensure each instance is independent.
                    AC_Hypers | dict - A dictioanry which may be unpacked and sent as arguments to AC init.
                    Env | gym.env - An optional copy of an envrionment to use. If it is omitted the env will be instanced within the function.
        '''

        np.random.seed(Input['Seed'])

        if 'Env' in Input.keys():
            Env = Input['Env']
        else:
            Env = gym.make(self.Env_String)
            Env.Set_Params(**self.Env_Params)
            if self.Over_Sample_Mult != 0 : Env.Over_Sample(Mult = self.Over_Sample_Mult, N_ = 5)

        AC = A2C.Actor_Critic(Environment = Env, **(Input['AC_Hypers']))
        AC.Train(self.N_Eps)

        Merton_Results, Agent_Results = Env.Validate(10000, AC)

        return Agent_Results['Sharpe'], Agent_Results['Mean_Utility'] - Merton_Results['Mean_Utility']
