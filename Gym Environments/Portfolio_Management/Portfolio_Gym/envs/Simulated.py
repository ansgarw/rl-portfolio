import numpy as np
import warnings
import gym
from scipy.optimize import minimize
import os
import pickle

'''
Required changes;
    1. Generate a cache of returns when the envrionment is instanced, and then just run through them.
    2. Generate these returns on a JIT basis, to minimise the likelihood of having to generate twice.
'''

class SimulatedEnv(gym.Env):

    def __init__(self, **kwargs):
        '''
        Initialise the environment with default settings.
        gym.make no longer supports arguments, hence the user cannot over-ride parameters here, and instead should call .Set_Params()
        '''

        self.Mu            = kwargs['Mu']
        self.Sigma         = kwargs['Sigma']
        self.Row           = kwargs['Row']
        self.Risk_Aversion = kwargs['Risk_Aversion']
        self.Rf            = kwargs['Rf']
        self.Min_Leverage  = kwargs['Min_Leverage']
        self.Max_Leverage  = kwargs['Max_Leverage']
        self.Time_Horizon  = kwargs['Time_Horizon']
        self.Time_Step     = kwargs['Time_Step']
        self.Intermediate_Reward = kwargs['Intermediate_Reward']

        self.Wealth = np.random.uniform(0.5, 1.5)
        self.Tau    = self.Time_Horizon

        self.Done   = False
        self.Reward = 0

        self.Mkt_Return = None

        self.Internal_Setup()


    def Set_Params (self, **kwargs):
        '''
        It is no longer possible to send arguments to gym.make(), so a second function call to this method is required to override the default parameters of the environment.

        Parameters
        ----------
            Mu | float, np.array, list
                The mean return of the asset(s). May be passed as either a float to run the environment with a single asset, or a numpy array or list to instance the environment with multiple assets.

            Sigma | float, np.array, list
                The standard deviation of the asset(s). Must have the same lenght as Mu.

            Row | float, np.array, list
                The corolation between assets, if mutiple assets are used. If only two assets are used it may be passed as a float, otherwise a square corolation matrix must be passed.

            Risk_Aversion | float
                The investors risk aversion, which is used to generate utility

            Rf | float
                The risk free rate

            Min_Leverage | float
                The smallest leverage which the agent may take. Must be negative to allow the agent too short.

            Max_Leverage | float
                The highest leverage that the agent may take

            Time_Horizon | float
                The investment horizon of an episode (Years)

            Time_Step | float
                The length of a single step (Years)

            Intermediate_Rewards | bool
                A flag indicating whether the environment will return intermediate rewards. Note for this parameter to have an effect Risk_Aversion must equal one, or it will default to false. Intermediate_Reward are calculated as the increase in utility across the step.


        Notes
        -----
            When setting Intermediate_Reward = True be sure that Risk_Aversion == 1, or Intermediate_Reward will have no effect.
        '''

        self.Mu            = kwargs['Mu']            if 'Mu'            in kwargs.keys() else self.Mu
        self.Rf            = kwargs['Rf']            if 'Rf'            in kwargs.keys() else self.Rf
        self.Sigma         = kwargs['Sigma']         if 'Sigma'         in kwargs.keys() else self.Sigma
        self.Row           = kwargs['Row']           if 'Row'           in kwargs.keys() else self.Row
        self.Risk_Aversion = kwargs['Risk_Aversion'] if 'Risk_Aversion' in kwargs.keys() else self.Risk_Aversion
        self.Min_Leverage  = kwargs['Min_Leverage']  if 'Min_Leverage'  in kwargs.keys() else self.Min_Leverage
        self.Max_Leverage  = kwargs['Max_Leverage']  if 'Max_Leverage'  in kwargs.keys() else self.Max_Leverage
        self.Time_Horizon  = kwargs['Time_Horizon']  if 'Time_Horizon'  in kwargs.keys() else self.Time_Horizon
        self.Time_Step     = kwargs['Time_Step']     if 'Time_Step'     in kwargs.keys() else self.Time_Step
        self.Intermediate_Reward = kwargs['Intermediate_Reward'] if 'Intermediate_Reward' in kwargs.keys() else self.Intermediate_Reward


        if isinstance(self.Mu, list):
            self.Mu = np.array(self.Mu)

        if isinstance(self.Sigma, list):
            self.Sigma = np.array(self.Sigma)

        if isinstance(self.Row, float) and self.Mu.size == 2:
            self.Row = np.array([[1, self.Row], [self.Row, 1]])

        elif isinstance(self.Row, list):
            self.Row = np.array(self.Row)

        self.Internal_Setup()

        if self.Intermediate_Reward == True and self.Risk_Aversion != 1:
            warnings.warn('Risk_Aversion is not one, Intermediate_Reward is disabled.')

        for key in kwargs.keys():
            if not key in ('Mu', 'Sigma', 'Rf', 'Row', 'Risk_Aversion', 'Max_Leverage', 'Min_Leverage', 'Time_Horizon', 'Time_Step', 'Intermediate_Reward'):
                print("Keyword:", key, "not recognised.")


    def Internal_Setup (self):
        '''
        Set up some internal parameters after initialisation or after parameters are overwridden
        This function is called automatically and should not be called externally by the user.
        '''

        self.Training_Var    = self.Sigma ** 2
        self.Training_Mean   = self.Mu - self.Rf
        self.Training_Merton = self.Merton_Fraction()

        self.observation_space = gym.spaces.Box(low = np.array([0.0, 0.0]), high = np.array([self.Time_Horizon, 100]), dtype = np.float32)
        if isinstance(self.Mu, int) or isinstance(self.Mu, float):
            self.Return_Func     = self.Single_Asset_GBM
            self.action_space    = gym.spaces.Box(low = self.Min_Leverage, high = self.Max_Leverage, shape = (1,), dtype = np.float32)

        else:
            self.Return_Func  = self.Multi_Asset_GBM
            self.action_space = gym.spaces.Box(low = np.array([self.Min_Leverage] * self.Mu.size), high = np.array([self.Max_Leverage] * self.Mu.size), dtype = np.float32)


    def step(self, action):
        '''
        Steps the environment forward, this is the main point of interface between an agent and the environment.

        Parameters
        ----------
            action | np.array (1D)
                The array must have the same length as the number of assets in the environment. The Nth value in the array represents the fraction of ones wealth to invest in the Nth asset. Negative values may be sent to enter short positions, or values with abs() > 1 to leverage oneself.

        Returns
        -------
            A tuple of the following 4 things:

            1. Observation | np.array (1D)
                The observation space of this environment is always a numpy array of length two, with the following variables: [Current_Wealth, Tau]

            2. Reward | float
                The reward for the last action taken.

            3. Done | bool
                Indicates whether the current episode has ended or not. Once an episode has ended the envrionment must be reset before another step may be taken.

            4. Info | dict
                A dictionary which includes extra information about the state of the environment which is not included in the observation as it is not relevent to the agent. Currently this includes:
                    'Mkt-Rf' : The excess return of the market on the last step
                    'Rfree'  : The risk free rate.
        '''

        assert self.action_space.contains(action), "Action %r (of type %s) is not within the action space" % (action, type(action))
        assert self.Done != True, 'Action attempted after epsisode has ended.'

        Return = self.Return_Func(action)

        if self.Risk_Aversion == 1 and self.Intermediate_Reward == True:
            self.Reward = self.Utility(self.Wealth * Return) - self.Utility()
        else:
            self.Reward = 0

        self.Wealth *= Return
        self.Tau    -= self.Time_Step

        self.Done  = False
        if self.Tau <= 0 or self.Wealth <= 0:
            self.Done = True
            self.Tau = 0 if self.Tau < 0 else self.Tau
            self.Reward = self.Utility()

        return self.Gen_State(), self.Reward, self.Done, self.Gen_Info()


    def reset(self):
        '''
        Resets the environment so a new episode may be ran. Must be called by the user / agent to begin an episode.

        Returns
        -------
            Oberservation | np.array (1D)
                The observation space of this environment is always a numpy array of length two, with the following variables: [Current_Wealth, Tau]

        Notes
        -----
            Wealth is initalised as a uniform random variable about 1 as that is the range across which the utlity curve's gradient variaes the most, and a random starting wealth helps the agents to experiance many wealths, and hence better map the value function.
        '''
        # Must return an intial Observation
        self.Wealth = np.random.uniform(0.5, 1.5)
        self.Tau    = self.Time_Horizon

        self.Done = False
        self.Reward = 0

        return self.Gen_State()


    def render(self):
        ''' Prints the current wealth and time horizon '''
        print("Current Wealth: " + str(round(self.Wealth, 4)) + ", Tau: " + str(self.Tau) + "\n")


    def Gen_State (self):
        '''
        Generates an observation (For internal use only)

        Returns
        -------
            np.array (1D)
                An observation
        '''
        return np.array([self.Wealth, self.Tau])


    def Gen_Info (self):
        '''
        Info contains information external to the state which may be used during validation to provide a benchmark to check
        the agent against. (For internal use only)

        Returns
        -------
            A dictionary with the following keys:
                    'Rfree'  : The current risk free rate
                    'Mkt-Rf' : The market excess return on the step
        '''

        Info = {'Rfree'  : self.Rf,
                'Mkt-Rf' : self.Mkt_Return}
        return Info


    def Single_Asset_GBM (self, Action):
        '''
        Randomly generates a return on a single asset, per a GBM (For internal use only)

        Parameters
        ----------
            Action | np.array (1D)
                A 1D array of length 1, which indicates the current leverage of the agent.

        Returns
        -------
            float
                1 + The decimal return generated across the period by the specified investment.

        '''

        # Generate returns from a normal distribution
        Mean = (self.Mu - (self.Sigma ** 2) / 2) * self.Time_Step
        Std = self.Sigma * (self.Time_Step ** 0.5)
        self.Mkt_Return = (np.exp(np.random.normal(Mean, Std)) - 1) - (self.Rf * self.Time_Step)
        return (1 + (self.Rf * self.Time_Step) + (Action[0] * self.Mkt_Return))


    def Multi_Asset_GBM(self, Action):
        '''
        Randomly generates returns across multiple assets, per a GBM (For internal use only)

        Parameters
        ----------
            Action | np.array (1D)
                A 1D array of length N, in which the Nth value in the array represents the fraction of ones wealth to invest in the Nth asset

        Returns
        -------
            float
                1 + The decimal return generated across the period by the specified investment.

        '''

        # Genrate the means and standard deviations
        Means = (self.Mu - (self.Sigma ** 2) / 2) * self.Time_Step
        Stds  = (self.Sigma * (self.Time_Step ** 0.5)).reshape(-1,1)
        Covs  = self.Row * np.matmul(Stds, Stds.T)

        self.Mkt_Return = (np.exp(np.random.multivariate_normal(Means, Covs)) - 1) - (self.Rf * self.Time_Step)
        return (1 + (self.Rf * self.Time_Step) + np.sum(Action * self.Mkt_Return))


    def Utility (self, *args):
        '''
        Calculates investor utility

        Parameters
        ----------
            None
                If no arguments are passed then the function returns the utility of the environments internal wealth.

            float (Optional)
                If a float is passed then the fucntion returns the utility using this value as wealth

            np.array (Optional)
                If a np array is passed then the function returns a np array of the same size, with each value replaced by the utility of the input value at the same index.

        Returns
        -------
            The utility either as a float if no input is given, or as the same type as the argument.

        Notes
        -----
            Since utility is undefined if wealth equals 0 or is negative, in this event a utility of -10 is returned.
        '''

        if len(args) == 1:
            Wealth = args[0]
        else:
            Wealth = self.Wealth

        if np.any(Wealth <= 0):
            return -10
        elif self.Risk_Aversion == 1:
            return np.log(Wealth)
        else:
            return (Wealth ** (1 - self.Risk_Aversion)) / (1 - self.Risk_Aversion)


    def Merton_Fraction (self):
        '''
        Calculates the merton portfolio fraction if there is only one asset, or the Markowitz portfolio and Merton portfolio fraction if multiple assets are being used.

        Returns
        -------
            float
                In the case of a single asset. Represents the Merton Portfolio Fraction

            np.array (1D)
                In the case of multiple assets. Represents the optimal fraction of ones wealth to invest in each asset.

        '''

        if isinstance(self.Mu, int) or isinstance(self.Mu, float):
            return (self.Training_Mean) / ((self.Training_Var) * self.Risk_Aversion)

        else:

            Std = self.Sigma.reshape(-1,1)
            Cov = self.Row * np.matmul(Std, Std.T)

            Data = {'Mean'  : self.Training_Mean,
                    'Var'   : self.Training_Var,
                    'Cov'   : Cov}

            cons = [{'type': 'ineq', 'fun': lambda x:  np.sum(x) - 1},
                    {'type': 'ineq', 'fun': lambda x: -np.sum(x) + 1}]

            Weights = np.array(minimize(Sharpe_Ratio, [1 / Data['Mean'].size] * Data['Mean'].size, args = (Data), constraints = cons).x).reshape(-1,1)
            Var = np.matmul(np.matmul(Weights.T, Cov), Weights)[0,0]
            Merton_Leverage = (np.sum(Weights * Data['Mean'])) / (self.Risk_Aversion * Var)

            return Weights * Merton_Leverage





class VAR_Engine:

    '''
    This model is capable of generating the returns of N corolated assets as well as M autoregressive factors, upon which the assets have linear dependence.
    The input is a little complex and best explained through an example:



        Example 1 :
            A single asset and a single factor. The asset return at time t+1 is dependent upon the factor at time t and t-1. The factor at time t+1 is dependent upon the factor at time t. Under these settings the model becomes:

            ln(1 + Return[t+1]) = Asset_Beta[0,0] + Asset_Beta[0,1] * Factor_1[t] + Asset_Beta[0,2] * Factor_1[t-1] + Epsilon[0]
            Factor_1[t+1] = Factor_Beta[0,0] + Factor_Beta[0,1] * Factor_1[t] + Epsilon[1]

            where:
                Epsilon = MVN(0, Cov)



        Example 2:
            A single asset with two factors. The asset return at time t+1 is dependent upon both factors at time t and t-1. Each factor at time t+1 is dependent upon itself at time t and t-1. Under these settings the model becomes:

            ln(1 + Return[t+1]) = Asset_Beta[0,0] + (Asset_Beta[0,1] * Factor_1[t]) + (Asset_Beta[0,2] * Factor_2[t]) + (Asset_Beta[0,3] * Factor_1[t-1]) + (Asset_Beta[0,4] * Factor_2[t-1]) + Epsilon[0]
            Factor_1[t+1] = Factor_Beta[0,0] + (Factor_Beta[0,1] * Factor_1[t]) + (Factor_Beta[0,2] * Factor_1[t-1]) + Epsilon[1]
            Factor_2[t+1] = Factor_Beta[1,0] + (Factor_Beta[1,1] * Factor_2[t]) + (Factor_Beta[1,2] * Factor_2[t-1]) + Epsilon[1]



        To include another asset, simple add another column to Asset_Beta, using the same structure as before. The model will automatically calculate how many periods of factors to include when generating the factors and asset returns based upon the length of the Asset_Beta and Factor_Beta inputs.





        Since generating setups for this model can be complex the following presets can be used:

        Brandt Market and dividend to price ratio
            'Factor_Beta' : numpy.array([-0.1694, 0.9514]).reshape(-1,1)
            'Asset_Beta'  : numpy.array([0.2049, 0.0568]).reshape(-1,1)
            'Cov'         : numpy.array([[6.225, -6.044], [-6.044, 6.316]]) * 1e-3
            'Period'      : 0.25


        Single asset single factor with high R2 (Should allow more dramatic outperformance of the RL Agents)
            'Factor_Beta' : np.array([-0.1694, 0.9514]).reshape(-1,1)
            'Asset_Beta'  : np.array([0.5549, 0.1568]).reshape(-1,1)
            'Cov'         : np.array([[6.225, -6.044], [-6.044, 6.316]]) * 1e-3
            'Period'      : 0.25



    '''

    def __init__ (self, Factor_Beta, Asset_Beta, Cov, Period):

        '''
        Parameters
        ----------
            Factor_Beta | np.array (2D)
                The Autoregressive betas of the factor(s), see examples above for structure.

            Asset_Beta | np.array (2D)
                The linear dependence of the asset(s) upon the last N values of the factor(s). See examples for structure.

            Cov | np.array (2D, Square)
                The covariance matrix of the Assets and the Factors.

            Period | float
                The time_step between subsequent returns.
        '''

        self.Factor_Beta = Factor_Beta
        self.Asset_Beta  = Asset_Beta
        self.Cov         = Cov
        self.Period      = Period

        # First calculate some problem dimensions
        self.Num_Assets    = self.Asset_Beta.shape[1]
        self.Num_Factors   = self.Factor_Beta.shape[1]
        self.Asset_AR_len  = int((self.Asset_Beta.shape[0] - 1) / self.Num_Factors)
        self.Factor_AR_len = self.Factor_Beta.shape[0] - 1

        if os.path.exists(FilePath + "Sim_Cache.pkl"):
            with open(FilePath + 'Sim_Cache.pkl', 'rb') as file:
                 self.Preset_Cache = pickle.load(file)
        else:
            self.Preset_Cache = {}

        assert self.Asset_Beta.shape[0] == 1 + self.Asset_AR_len * self.Num_Factors, 'Ensure that the asset corrolates to each factor for the same number of periods'
        assert self.Cov.shape[0] == self.Num_Assets + self.Num_Factors, 'Cov dimensions do not match the number of assets and factors'


    def reset (self):
        '''
        Reset the model.
        Generate enough preiods values for the factors to begin calculating the Asset return including their dependence. Factors are initiated at their stationary point + a small amount of noise.
        '''

        self.Factor_Values = np.ones((max(self.Factor_AR_len, self.Asset_AR_len), self.Num_Factors))
        for i in range(self.Factor_Values.shape[1]):
            self.Factor_Values[:,i] *= (self.Factor_Beta[0,i] / (1 - np.sum(self.Factor_Beta[1:,i]))) + np.random.normal(0, self.Cov[i,i] ** 0.5)


    def step (self):
        '''
        Step the model forward by one period.

        Returns
        -------
            np.array (1D)
                The returns of the assets across the period.
        '''

        if not hasattr(self, 'Factor_Values') : self.reset()

        Epsilon = np.random.multivariate_normal(np.zeros(self.Cov.shape[0]), self.Cov)

        New_Factors = self.Factor_Beta[0] + np.sum(self.Factor_Beta[1:] * self.Factor_Values[-self.Factor_AR_len:][::-1], axis = 0) + Epsilon[self.Num_Assets:]

        Return_X = self.Factor_Values[-self.Asset_AR_len:][::-1].reshape(-1,1)
        for _ in range(self.Num_Assets-1):
            Return_X = np.hstack((Return_X, Return_X[:,0].reshape(-1,1)))
        New_Returns = self.Asset_Beta[0] + np.sum(self.Asset_Beta[1:] * Return_X, axis = 0) + Epsilon[:self.Num_Assets]

        self.Factor_Values = np.vstack((self.Factor_Values, New_Factors.reshape(1,-1)))

        return np.exp(New_Returns) - 1


    def Genrate_Returns (self, N, State_Hist_Len = 1):
        '''

        Parameters
        ----------
            N | int
                The number of periods of returns to generate.

            State_Hist_Len | int
                The number of periods of factors to include in the state.

        Returns
        -------
            A tuple containing the following data:

            1. np.array (2D)
                Asset returns for N consecutive periods

            2. np.array (2D)
                The values of the factors for N consecutive periods

        Notes
        -----
            Due to the possible complexity of this model, it is necessary to calculate the moments of the asset(s) numerically.

        '''

        self.reset()
        Returns = np.zeros((N, self.Num_Assets))
        Factors = np.zeros((N, self.Num_Factors * State_Hist_Len))

        for i in range(200000):
            Returns[i] = self.step().flatten()
            Factors[i] = self.Factor_Values[-State_Hist_Len:][::-1].flatten()

        self.reset()

        return Returns, Factors





class Simulated_VAR_Env(gym.Env):

    def __init__ (self, **kwargs):
        '''
        Initialise the environment with default settings
        gym.make no longer accepts user agruments, hence Set_Params should be called to overwrite default settings.
        '''

        self.VAR_Model = VAR_Engine(kwargs['Factor_Beta'], kwargs['Asset_Beta'], kwargs['Cov'], kwargs['Period'])

        self.Risk_Aversion = kwargs['Risk_Aversion']
        self.Rf            = kwargs['Rf']
        self.Min_Leverage  = kwargs['Min_Leverage']
        self.Max_Leverage  = kwargs['Max_Leverage']
        self.Episode_Length      = kwargs['Episode_Length']
        self.Intermediate_Reward = kwargs['Intermediate_Reward']
        self.Factor_State_Len    = kwargs['Factor_State_Len']
        self.Num_Returns         = 200000

        self.Wealth = np.clip(np.random.normal(1, 0.25), 0.25, 1.75)
        self.Tau    = self.Episode_Length * self.VAR_Model.Period

        self.Done   = False
        self.Reward = 0

        self.observation_space = gym.spaces.Box(low = np.array([0.0, 0.0] + [-100] * self.Factor_State_Len * self.VAR_Model.Num_Factors), high = np.array([self.Tau, 100] + [100] * self.Factor_State_Len * self.VAR_Model.Num_Factors), dtype = np.float32)
        self.action_space      = gym.spaces.Box(low = np.array([self.Min_Leverage] * self.VAR_Model.Num_Assets), high = np.array([self.Max_Leverage] * self.VAR_Model.Num_Assets), dtype = np.float32)


    def Set_Params(self, **kwargs):
        '''
        It is no longer possible to send arguments to gym.make(), so a second function call to this method is required to override the default parameters of the environment.

        Parameters
        ----------
            Factor_Beta | np.array (2D)
                VAR_Engine Factor_Beta.

            Asset_Beta | np.array (2D)
                VAR_Engine Asset_Beta.

            Cov | np.array (2D)
                VAR_Engine Cov.

            Period | np.array (2D)
                VAR_Engine Period.

            Risk_Aversion | float
                The investors risk aversion, which is used to generate utility

            Rf | float
                The risk free rate

            Min_Leverage | float
                The smallest leverage which the agent may take. Must be negative to allow the agent too short.

            Max_Leverage | float
                The highest leverage that the agent may take

            Episode_Length | int
                The number of steps that an episode should consist of. (Time horizon and step are no longer used as period is a property of the underlying VAR model.)

            Intermediate_Rewards | bool
                A flag indicating whether the environment will return intermediate rewards. Note for this parameter to have an effect Risk_Aversion must equal one, or it will default to false. Intermediate_Reward are calculated as the increase in utility across the step.

            Factor_State_Len | int
                The number of periods of historical factor values to include in the state.

            Num_Returns | int
                The lenght of the return database from which experiance is sampled, in steps.


        Notes
        -----
            When setting Intermediate_Reward = True be sure that Risk_Aversion == 1, or Intermediate_Reward will have no effect.
            When re-specifying the underlying VAR model, all VAR_Model parameters must be specified.
        '''

        if set(['Factor_Beta', 'Asset_Beta', 'Cov', 'Period']).issubset(set(kwargs.keys())):
            self.VAR_Model = VAR_Engine(kwargs['Factor_Beta'], kwargs['Asset_Beta'], kwargs['Cov'], kwargs['Period'])


        self.Risk_Aversion       = kwargs['Risk_Aversion']       if 'Risk_Aversion'       in kwargs.keys() else self.Risk_Aversion
        self.Rf                  = kwargs['Rf']                  if 'Rf'                  in kwargs.keys() else self.Rf
        self.Min_Leverage        = kwargs['Min_Leverage']        if 'Min_Leverage'        in kwargs.keys() else self.Min_Leverage
        self.Max_Leverage        = kwargs['Max_Leverage']        if 'Max_Leverage'        in kwargs.keys() else self.Max_Leverage
        self.Episode_Length      = kwargs['Episode_Length']      if 'Episode_Length'      in kwargs.keys() else self.Episode_Length
        self.Intermediate_Reward = kwargs['Intermediate_Reward'] if 'Intermediate_Reward' in kwargs.keys() else self.Intermediate_Reward
        self.Factor_State_Len    = kwargs['Factor_State_Len']    if 'Factor_State_Len'    in kwargs.keys() else self.Factor_State_Len
        self.Num_Returns         = kwargs['Num_Returns']         if 'Num_Returns'         in kwargs.keys() else self.Num_Returns


        self.observation_space = gym.spaces.Box(low = np.array([0.0, 0.0] + [-100] * self.Factor_State_Len * self.VAR_Model.Num_Factors), high = np.array([self.VAR_Model.Period * self.Episode_Length, 100] + [100] * self.Factor_State_Len * self.VAR_Model.Num_Factors), dtype = np.float32)
        self.action_space      = gym.spaces.Box(low = np.array([self.Min_Leverage] * self.VAR_Model.Num_Assets), high = np.array([self.Max_Leverage] * self.VAR_Model.Num_Assets), dtype = np.float32)

        if self.Intermediate_Reward == True and self.Risk_Aversion != 1:
            warnings.warn('Risk_Aversion is not one, Intermediate_Reward is disabled.')

        self.Genrate_Returns()


    def Genrate_Returns (self):

        self.Returns, self.Factors = self.VAR_Model.Genrate_Returns(self.Num_Returns, self.Factor_State_Len)

        self.Training_Mean = np.mean(self.Returns, axis = 0)
        self.Training_Var  = np.var(self.Returns, axis = 0)
        self.Training_Cov  = np.cov(self.Returns, rowvar = False)
        self.Training_Merton = self.Merton_Fraction()


    def reset (self):
        '''
        Resets the environment so a new episode may be ran. Must be called by the user / agent to begin an episode.

        Returns
        -------
            Oberservation | np.array (1D)
                The oberservation space of this environment will include [Wealth, Tau] as well as the specified number of historical factor values.

        Notes
        -----
            Wealth is initalised as a uniform random variable about 1 as that is the range across which the utlity curve's gradient variaes the most, and a random starting wealth helps the agents to experiance many wealths, and hence better map the value function.
        '''

        if not hasattr(self, 'Returns'):
            self.Genrate_Returns()

        self.Index = np.random.randint(low = 0, high = self.Returns.shape[0] - self.Episode_Length)

        self.Wealth = np.clip(np.random.normal(1, 0.25), 0.25, 1.75)
        self.Tau    = self.Episode_Length * self.VAR_Model.Period

        self.Done   = False
        self.Reward = 0

        return self.Gen_State()


    def step (self, Action):
        '''
        Steps the environment forward, this is the main point of interface between an agent and the environment.

        Parameters
        ----------
            action | np.array (1D)
                The array must have the same length as the number of assets in the environment. The Nth value in the array represents the fraction of ones wealth to invest in the Nth asset. Negative values may be sent to enter short positions, or values with abs() > 1 to leverage oneself.

        Returns
        -------
            A tuple of the following 4 things:

            1. Observation | np.array (1D)
                The oberservation space of this environment will include [Wealth, Tau] as well as the specified number of historical factor values.

            2. Reward | float
                The reward for the last action taken.

            3. Done | bool
                Indicates whether the current episode has ended or not. Once an episode has ended the envrionment must be reset before another step may be taken.

            4. Info | dict
                A dictionary which includes extra information about the state of the environment which is not included in the observation as it is not relevent to the agent. Currently this includes:
                    'Mkt-Rf' : The excess return of the market on the last step
                    'Rfree'  : The risk free rate.
        '''

        assert self.action_space.contains(Action), "Action %r (of type %s) is not within the action space." % (Action, type(Action))
        assert self.Done != True, 'Action attempted after epsisode has ended.'

        self.Index += 1
        Investment_Return = (1 + (self.Rf * self.VAR_Model.Period) + np.sum(Action * self.Returns[self.Index]))

        if self.Risk_Aversion == 1 and self.Intermediate_Reward == True:
            self.Reward = self.Utility(self.Wealth * Investment_Return) - self.Utility(self.Wealth)
        else:
            self.Reward = 0

        self.Wealth *= Investment_Return
        self.Tau    -= self.VAR_Model.Period

        self.Done  = False
        if self.Tau <= 0 or self.Wealth <= 0:
            self.Done = True
            self.Tau = 0 if self.Tau < 0 else self.Tau
            self.Reward = self.Utility()

        return self.Gen_State(), self.Reward, self.Done, self.Gen_Info()


    def render (self):
        ''' Prints the current wealth and time horizon '''
        print("Current Wealth: " + str(round(self.Wealth, 4)) + ", Tau: " + str(self.Tau) + "\n")


    def Gen_State (self):
        '''
        Generates an observation (For internal use only)

        Returns
        -------
            np.array (1D)
                An observation
        '''
        return np.append(np.array([self.Wealth, self.Tau]), self.Factors[self.Index])


    def Gen_Info (self):
        '''
        Info contains information external to the state which may be used during validation to provide a benchmark to check
        the agent against.

        Returns
        -------
            A dictionary with the following keys:
                    'Rfree'  : The current risk free rate
                    'Mkt-Rf' : The market excess return on the step
        '''

        Info = {'Rfree'  : self.Rf,
                'Mkt-Rf' : self.Returns[self.Index]}

        return Info


    def Utility (self, *args):
        '''
        Calculates investor utility

        Parameters
        ----------
            None
                If no arguments are passed then the function returns the utility of the environments internal wealth.

            float (Optional)
                If a float is passed then the fucntion returns the utility using this value as wealth

            np.array (Optional)
                If a np array is passed then the function returns a np array of the same size, with each value replaced by the utility of the input value at the same index.

        Returns
        -------
            The utility either as a float if no input is given, or as the same type as the argument.

        Notes
        -----
            Since utility is undefined if wealth equals 0 or is negative, in this event a utility of -10 is returned.
        '''

        if len(args) == 1:
            Wealth = args[0]
        else:
            Wealth = self.Wealth

        if np.any(Wealth <= 0):
            return -10
        elif self.Risk_Aversion == 1:
            return np.log(Wealth)
        else:
            return (Wealth ** (1 - self.Risk_Aversion)) / (1 - self.Risk_Aversion)


    def Merton_Fraction (self):
        '''
        Calculates the merton portfolio fraction if there is only one asset, or the Markowitz portfolio and Merton portfolio fraction if multiple assets are being used.

        Returns
        -------
            float
                In the case of a single asset. Represents the Merton Portfolio Fraction

            np.array (1D)
                In the case of multiple assets. Represents the optimal fraction of ones wealth to invest in each asset.

        '''

        if self.VAR_Model.Num_Assets == 1:
            return self.Training_Mean[0] / ((self.Training_Var[0]) * self.Risk_Aversion)

        else:
            Data = {'Mean'  : self.Training_Mean,
                    'Var'   : self.Training_Var,
                    'Cov'   : self.Training_Cov}

            cons = [{'type': 'ineq', 'fun': lambda x:  np.sum(x) - 1},
                    {'type': 'ineq', 'fun': lambda x: -np.sum(x) + 1}]

            Weights = np.array(minimize(Sharpe_Ratio, [1 / Data['Mean'].size] * Data['Mean'].size, args = (Data), constraints = cons).x).reshape(-1,1)
            Var = np.matmul(np.matmul(Weights.T, self.Training_Cov), Weights)[0,0]
            Merton_Leverage = (np.sum(Weights * Data['Mean'])) / (self.Risk_Aversion * Var)

            return Weights * Merton_Leverage



def Sharpe_Ratio(Weights, Data):
    '''
    This function is used by the Merton_Fraction to calculate the Markowitz portfolio.

    Parameters
    ----------
        Data | dict
            A dictionary with the following keys:

            'Mean' | np.array 1D
                The mean excess returns of the asset(s)

            'Var' | np.array 1D
                The variance of return of the asset(s)

            'Cov' | np.array 2D
                The covaraince matrix of asset(s) return

    Returns
    -------
        float
            The negative absolute sharpe ratio.

    Notes
    -----
        Scipy.optimise.minimise is used for the optimiser routine, hence the negative sharpe must be returned.
        The abs sharpe is used, as we can just short the portfolio with the highest abs sharpe if it happens to be negative. This ensures the sim works with assets with both positive and negative mean returns.
    '''

    Weights = np.array(Weights).reshape(-1,1)
    Excess_ret = np.sum(Weights * Data['Mean'])

    Var = np.matmul(np.matmul(Weights.T, Data['Cov']), Weights)

    Sharpe = Excess_ret / (Var ** 0.5)[0,0]

    return -np.abs(Sharpe)
