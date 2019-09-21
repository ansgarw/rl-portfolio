import numpy as np
import warnings
import gym

# CHANGELOG
# Changed max wealth in obs. space from inf to 100 fold.
# Adjusted default values for Max/Min leverage as well as Time_Step

# New features to support AC
# Initial wealth is uniformly distributed between 0 and 1
# Tau is now in the units of years rather than steps
# Episode is ended if negative wealth is observed
# Power_Utility updated to handle negative wealth with a constant penalisation
# Instantaneous reward is now optional if risk aversion is one.


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
        # if MODEL == SOME_OTHER_MODEL ...

        self.Wealth = np.random.uniform()
        self.Tau    = self.Time_Horizon
        self.State  = np.array([self.Tau, self.Wealth])

        self.Done   = False
        self.Reward = 0

        # 0: Tau
        # 1: Wealth
        self.observation_space = gym.spaces.Box(low = np.array([0.0, 0.0]), high = np.array([self.Time_Horizon, 100]), dtype = np.float32)

        if isinstance(self.Mu, int) or isinstance(self.Mu, float):
            self.Return_Func = self.Single_Asset_GBM
            self.action_space = gym.spaces.Box(low = self.Min_Leverage, high = self.Max_Leverage, shape = (1,), dtype = np.float32)
        else:
            self.Return_Func = self.Multi_Asset_GBM
            self.action_space = gym.spaces.Box(low = np.array([self.Min_Leverage] * self.Mu.size), high = np.array([self.Max_Leverage] * self.Mu.size), dtype = np.float32)


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
            Model                : The model to use to generate returns. Currently the only acceptable argument is “GBM”.
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

        # Update action space
        if isinstance(self.Mu, int) or isinstance(self.Mu, float):
            assert isinstance(self.Sigma, int) or isinstance(self.Sigma, float), "Sigma type does not match Mu"
            self.Return_Func = self.Single_Asset_GBM
            self.action_space = gym.spaces.Box(low = self.Min_Leverage, high = self.Max_Leverage, shape = (1,), dtype = np.float32)
        else:
            assert self.Mu.size == self.Sigma.size, "The size of Mu and Sigma must match"
            self.Return_Func = self.Multi_Asset_GBM
            self.action_space = gym.spaces.Box(low = np.array([self.Min_Leverage] * self.Mu.size), high = np.array([self.Max_Leverage] * self.Mu.size), dtype = np.float32)

        for key in kwargs.keys():
            if not key in ('Mu', 'Sigma', 'Rf', 'Row', 'Risk_Aversion', 'Max_Leverage', 'Min_Leverage', 'Time_Horizon', 'Time_Step', 'Intermediate_Reward'):
                print("Keyword:", key, "not recognised.")


    def step(self, action):
        assert self.action_space.contains(action), "Action %r (of type %s) is not within the action space" % (action, type(action))

        if self.Done == True:
            warnings.warn("Action attempted when episode is over.", Warning)

        if self.Risk_Aversion == 1 and self.Intermediate_Reward == True:
            self.Reward = self.Utility()

        self.Wealth *= self.Return_Func(action)
        self.Tau    -= self.Time_Step

        if self.Risk_Aversion == 1 and self.Intermediate_Reward == True:
            self.Reward = (self.Utility() - self.Reward)
        else:
            self.Reward = 0

        self.Done  = False
        if self.Tau <= 0 or self.Wealth <= 0:
            self.Done = True
            self.Tau = 0 if self.Tau < 0 else self.Tau
            self.Reward = self.Utility()

        self.State = np.array([self.Tau, self.Wealth])

        return self.State, self.Reward, self.Done, {}


    def reset(self):
        # Must return an intial Observation
        self.Wealth = np.random.uniform()
        self.Tau    = self.Time_Horizon
        self.State  = np.array([self.Tau, self.Wealth])

        self.Done = False
        self.Reward = 0

        return self.State


    def render(self):
        print("Current Wealth: " + str(round(self.Wealth, 4)) + ", Tau: " + str(self.Tau) + "\n")





    def Single_Asset_GBM(self, Action):
        # Generate returns from a normal distribution
        Mean = (self.Mu - (self.Sigma ** 2) / 2) * self.Time_Step
        Std = self.Sigma * (self.Time_Step ** 0.5)
        Return = (np.exp(np.random.normal(Mean, Std)) - 1) - (self.Rf * self.Time_Step)
        Net_Return = (1 + (self.Rf * self.Time_Step) + (Action[0] * Return))

        return Net_Return


    def Multi_Asset_GBM(self, Action):
        # Genrate the means and standard deviations
        Means = (self.Mu - (self.Sigma ** 2) / 2) * self.Time_Step
        Stds  = (self.Sigma * (self.Time_Step ** 0.5)).reshape(-1,1)
        Covs  = self.Row * np.matmul(Stds, Stds.T)

        Returns = (np.exp(np.random.multivariate_normal(Means, Covs)) - 1) - (self.Rf * self.Time_Step)
        Net_Return = (1 + (self.Rf * self.Time_Step) + np.sum(Action * Returns))

        return Net_Return


    def Utility(self):
        if self.Wealth <= 0:
            return -10

        if self.Risk_Aversion == 1:
            return np.log(self.Wealth)

        else:
            return (self.Wealth ** (1 - self.Risk_Aversion)) / (1 - self.Risk_Aversion)
