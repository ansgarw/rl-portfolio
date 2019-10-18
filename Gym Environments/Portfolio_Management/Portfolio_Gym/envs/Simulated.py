import numpy as np
import warnings
import gym
from scipy.optimize import minimize





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

        self.Wealth = np.clip(np.random.normal(1, 0.25), 0.25, 1.75)
        self.Tau    = self.Time_Horizon

        self.Done   = False
        self.Reward = 0

        self.Mkt_Return = None

        self.Internal_Setup()


    def Set_Params (self, **kwargs):
        '''
        Unfortunately it is no longer possible to send arguments to gym.make (as of 19/9/19), so this method may be used to
        change the Parameters of the environment from their defaults.

        Parameters
        ----------
            Mu                   : The mean return of the asset. May be passes as either a float if you wish to run the environment for a single asset,
                                   or a numpy array of Mus to instance the    environment with multiple assets.
            Sigma                : The standard deviation of assets returns. Must have the same type and shape as Mu.
            Row                  : A correlation matrix for the assets. Must be square and symmetrical. May be ignored if only one asset is used.
            Risk_Aversion        : The agents risk aversion, used only when calculating utility (which is the reward function).
            Rf                   : The risk free rate.
            Min_Leverage         : The smallest leverage which the agent may take. Must be negative to allow the agent too short.
            Max_Leverage         : The highest leverage that the agent may take
            Time_Horizon         : The investment horizon of an episode (Years)
            Time_Step            : The length of a single step (Years)
            Intermediate_Rewards : A flag indicating whether the environment will return intermediate rewards. Note for this parameter to have an
                                   effect Risk_Aversion must equal one, or it will default to false.
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
        self.Training_Var    = self.Sigma ** 2
        self.Training_Mean   = self.Mu
        self.Training_Merton = self.Merton_Fraction()

        self.observation_space = gym.spaces.Box(low = np.array([0.0, 0.0]), high = np.array([self.Time_Horizon, 100]), dtype = np.float32)
        if isinstance(self.Mu, int) or isinstance(self.Mu, float):
            self.Return_Func     = self.Single_Asset_GBM
            self.action_space    = gym.spaces.Box(low = self.Min_Leverage, high = self.Max_Leverage, shape = (1,), dtype = np.float32)

        else:
            self.Return_Func = self.Multi_Asset_GBM
            self.action_space = gym.spaces.Box(low = np.array([self.Min_Leverage] * self.Mu.size), high = np.array([self.Max_Leverage] * self.Mu.size), dtype = np.float32)


    def step(self, action):
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
        # Must return an intial Observation
        self.Wealth = np.clip(np.random.normal(1, 0.25), 0.25, 1.75)
        self.Tau    = self.Time_Horizon

        self.Done = False
        self.Reward = 0

        return self.Gen_State()


    def render(self):
        print("Current Wealth: " + str(round(self.Wealth, 4)) + ", Tau: " + str(self.Tau) + "\n")


    def Gen_State (self):
        return np.array([self.Wealth, self.Tau])


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
                'Mkt-Rf' : self.Mkt_Return}
        return Info


    def Single_Asset_GBM (self, Action):
        # Generate returns from a normal distribution
        Mean = (self.Mu - (self.Sigma ** 2) / 2) * self.Time_Step
        Std = self.Sigma * (self.Time_Step ** 0.5)
        self.Mkt_Return = (np.exp(np.random.normal(Mean, Std)) - 1) - (self.Rf * self.Time_Step)
        return (1 + (self.Rf * self.Time_Step) + (Action[0] * self.Mkt_Return))


    def Multi_Asset_GBM(self, Action):
        # Genrate the means and standard deviations
        Means = (self.Mu - (self.Sigma ** 2) / 2) * self.Time_Step
        Stds  = (self.Sigma * (self.Time_Step ** 0.5)).reshape(-1,1)
        Covs  = self.Row * np.matmul(Stds, Stds.T)

        self.Mkt_Return = (np.exp(np.random.multivariate_normal(Means, Covs)) - 1) - (self.Rf * self.Time_Step)
        return (1 + (self.Rf * self.Time_Step) + np.sum(Action * self.Mkt_Return))


    def Utility (self, *args):
        ''' Determine the utility of the investor at the end of life '''

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
        ''' Returns: The merton fraction of the current environment '''

        if isinstance(self.Mu, int) or isinstance(self.Mu, float):
            return (self.Training_Mean - self.Rf) / ((self.Training_Var) * self.Risk_Aversion)

        else:

            Std = self.Sigma.reshape(-1,1)
            Cov = self.Row * np.matmul(Std, Std.T)

            Data = {'Mean'  : self.Training_Mean - self.Rf,
                    'Var'   : self.Training_Var,
                    'Cov'   : Cov}

            cons = [{'type': 'ineq', 'fun': lambda x:  np.sum(x) - 1},
                    {'type': 'ineq', 'fun': lambda x: -np.sum(x) + 1}]

            Weights = np.array(minimize(Sharpe_Ratio, [1 / Data['Mean'].size] * Data['Mean'].size, args = (Data), constraints = cons).x).reshape(-1,1)
            Var = np.matmul(np.matmul(Weights.T, Cov), Weights)[0,0]
            Merton_Leverage = (np.sum(Weights * Data['Mean'])) / (self.Risk_Aversion * Var)

            return Weights * Merton_Leverage





class VAR_Engine:

    def __init__ (self, Factor_Beta, Asset_Beta, Cov):

        self.Factor_Beta = Factor_Beta
        self.Asset_Beta  = Asset_Beta
        self.Cov         = Cov

        # First calculate some problem dimensions
        self.Num_Assets    = self.Asset_Beta.shape[1]
        self.Num_Factors   = self.Factor_Beta.shape[1]
        self.Asset_AR_len  = int((self.Asset_Beta.shape[0] - 1) / self.Num_Factors)
        self.Factor_AR_len = self.Factor_Beta.shape[0] - 1

        assert self.Asset_Beta.shape[0] == 1 + self.Asset_AR_len * self.Num_Factors, 'Ensure that the asset corrolates to each factor for the same number of periods'
        assert self.Cov.shape[0] == self.Num_Assets + self.Num_Factors, 'Cov dimensions do not match the number of assets and factors'

    def reset (self):
        ''' Ensure that the factor matrix is long enough, and initiate each factor with its stationary value.'''
        self.Factor_Values = np.ones((max(self.Factor_AR_len, self.Asset_AR_len), self.Num_Factors))
        for i in range(self.Factor_Values.shape[1]):
            self.Factor_Values[:,i] *= self.Factor_Beta[0,i] / (1 - np.sum(self.Factor_Beta[1:,i]))

    def step (self):
        '''  '''
        Epsilon = np.random.multivariate_normal(np.zeros(self.Cov.shape[0]), self.Cov)

        New_Factors = self.Factor_Beta[0] + np.sum(self.Factor_Beta[1:] * self.Factor_Values[-self.Factor_AR_len:][::-1], axis = 0) + Epsilon[self.Num_Assets:]

        Return_X    = self.Factor_Values[-self.Asset_AR_len:][::-1].reshape(-1,1)
        for _ in range(self.Num_Assets-1):
            Return_X = np.hstack((Return_X, Return_X[:,0].reshape(-1,1)))
        New_Returns = self.Asset_Beta[0] + np.sum(self.Asset_Beta[1:] * Return_X, axis = 0) + Epsilon[:self.Num_Assets]

        self.Factor_Values = np.vstack((self.Factor_Values, New_Factors.reshape(1,-1)))

        return New_Returns

    def Moments (self):
        ''' Returns: The mean and standard deviation of the assets' returns '''

        # Use monte carlo simulation, as the variance of each asset becomes convoluted as the model size grows.
        self.reset()
        Returns = []
        for _ in range(100000):
            Returns.append(list(self.step()))

        self.reset()
        return np.mean(np.array(Returns), axis = 0), np.std(np.array(Returns), axis = 0), np.cov(np.array(Returns), rowvar = False)





class Simulated_VAR_Env(gym.Env):

    def __init__ (self, **kwargs):
        ''' Initialise the environment with default settings '''

        self.VAR_Model = VAR_Engine(kwargs['Factor_Beta'], kwargs['Asset_Beta'], kwargs['Cov'])

        self.Risk_Aversion = kwargs['Risk_Aversion']
        self.Rf            = kwargs['Rf']
        self.Min_Leverage  = kwargs['Min_Leverage']
        self.Max_Leverage  = kwargs['Max_Leverage']
        self.Time_Horizon  = kwargs['Time_Horizon']
        self.Time_Step     = kwargs['Time_Step']
        self.Intermediate_Reward = kwargs['Intermediate_Reward']
        self.Factor_State_Len    = kwargs['Factor_State_Len']

        self.Wealth = np.clip(np.random.normal(1, 0.25), 0.25, 1.75)
        self.Tau    = self.Time_Horizon

        self.Done   = False
        self.Reward = 0

        # Placeholder Parameters - Required by Wrapper
        self.Training_Mean, self.Training_Var, self.Training_Cov = self.VAR_Model.Moments()
        self.Training_Var = self.Training_Var ** 2
        self.Training_Merton = self.Merton_Fraction()
        self.Mkt_Return      = None

        self.observation_space = gym.spaces.Box(low = np.array([0.0, 0.0] + [-100] * self.Factor_State_Len * self.VAR_Model.Num_Factors), high = np.array([self.Time_Horizon, 100] + [100] * self.Factor_State_Len * self.VAR_Model.Num_Factors), dtype = np.float32)
        self.action_space      = gym.spaces.Box(low = np.array([self.Min_Leverage] * self.VAR_Model.Num_Assets), high = np.array([self.Max_Leverage] * self.VAR_Model.Num_Assets), dtype = np.float32)


    def Set_Params(self, **kwargs):

        if set(['Factor_Beta', 'Asset_Beta', 'Cov']).issubset(set(kwargs.keys())):
            self.VAR_Model = VAR_Engine(kwargs['Factor_Beta'], kwargs['Asset_Beta'], kwargs['Cov'])

        self.Risk_Aversion = kwargs['Risk_Aversion'] if 'Risk_Aversion' in kwargs.keys() else self.Risk_Aversion
        self.Rf            = kwargs['Rf']            if 'Rf'            in kwargs.keys() else self.Rf
        self.Min_Leverage  = kwargs['Min_Leverage']  if 'Min_Leverage'  in kwargs.keys() else self.Min_Leverage
        self.Max_Leverage  = kwargs['Max_Leverage']  if 'Max_Leverage'  in kwargs.keys() else self.Max_Leverage
        self.Time_Horizon  = kwargs['Time_Horizon']  if 'Time_Horizon'  in kwargs.keys() else self.Time_Horizon
        self.Time_Step     = kwargs['Time_Step']     if 'Time_Step'     in kwargs.keys() else self.Time_Step
        self.Intermediate_Reward = kwargs['Intermediate_Reward'] if 'Intermediate_Reward' in kwargs.keys() else self.Intermediate_Reward
        self.Factor_State_Len    = kwargs['Factor_State_Len']    if 'Factor_State_Len'    in kwargs.keys() else self.Factor_State_Len


        # Placeholder Parameters - Required by Wrapper
        self.Training_Mean, self.Training_Var, self.Training_Cov = self.VAR_Model.Moments()
        self.Training_Var = self.Training_Var ** 2
        self.Training_Merton = self.Merton_Fraction()
        self.Mkt_Return      = None

        self.observation_space = gym.spaces.Box(low = np.array([0.0, 0.0] + [-100] * self.Factor_State_Len * self.VAR_Model.Num_Factors), high = np.array([self.Time_Horizon, 100] + [100] * self.Factor_State_Len * self.VAR_Model.Num_Factors), dtype = np.float32)
        self.action_space      = gym.spaces.Box(low = np.array([self.Min_Leverage] * self.VAR_Model.Num_Assets), high = np.array([self.Max_Leverage] * self.VAR_Model.Num_Assets), dtype = np.float32)


    def reset (self):
        ''' Resets the environment '''

        self.VAR_Model.reset()

        self.Wealth = np.clip(np.random.normal(1, 0.25), 0.25, 1.75)
        self.Tau    = self.Time_Horizon

        self.Done   = False
        self.Reward = 0

        return self.Gen_State()


    def step (self, Action):
        assert self.action_space.contains(Action), "Action %r (of type %s) is not within the action space." % (Action, type(Action))
        assert self.Done != True, 'Action attempted after epsisode has ended.'

        self.Mkt_Return = self.VAR_Model.step()
        Investment_Return = (1 + (self.Rf * self.Time_Step) + np.sum(Action * self.Mkt_Return))

        if self.Risk_Aversion == 1 and self.Intermediate_Reward == True:
            self.Reward = self.Utility(self.Wealth * Investment_Return) - self.Utility(self.Wealth)
        else:
            self.Reward = 0

        self.Wealth *= Investment_Return
        self.Tau    -= self.Time_Step

        self.Done  = False
        if self.Tau <= 0 or self.Wealth <= 0:
            self.Done = True
            self.Tau = 0 if self.Tau < 0 else self.Tau
            self.Reward = self.Utility()

        return self.Gen_State(), self.Reward, self.Done, self.Gen_Info()


    def render (self):
        print("Current Wealth: " + str(round(self.Wealth, 4)) + ", Tau: " + str(self.Tau) + "\n")


    def Gen_State (self):
        ''' Generate and return a state obeservation '''
        return np.append(np.array([self.Wealth, self.Tau]), self.VAR_Model.Factor_Values[-self.Factor_State_Len:][::-1].flatten())


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
                'Mkt-Rf' : self.Mkt_Return}

        return Info


    def Utility (self, *args):
        ''' Determine the utility of the investor at the end of life '''

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
        ''' Returns: The merton fraction of the current environment '''

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

            return (np.sum(Weights * Data['Mean']) - self.Rf) / (self.Risk_Aversion * Var)





def Sharpe_Ratio(Weights, Data):
    Weights = np.array(Weights).reshape(-1,1)
    Excess_ret = np.sum(Weights * Data['Mean'])

    Var = np.matmul(np.matmul(Weights.T, Data['Cov']), Weights)

    Sharpe = Excess_ret / (Var ** 0.5)[0,0]

    return -Sharpe
