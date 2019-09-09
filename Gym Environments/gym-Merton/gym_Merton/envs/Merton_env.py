import gym
from gym import error, spaces, utils
from gym.utils import seeding
import warnings
import numpy as np
import os

# CHANGELOG
# Changed max wealth in obs. space from inf to 100 fold.
# Adjusted default values for Max/Min leverage as well as Time_Step

# New features to support AC
# Initial wealth is uniformly distributed between 0 and 1
# Tau is now in the units of years rather than steps
# Episode is ended if negative wealth is observed
# Power_Utility updated to handle negative wealth with a constant penalisation
# Instantaneous reward is now optional if risk aversion is one.

# directory = os.path.realpath(__file__).replace("Merton_env.py", "Helper.py")
# exec(open(directory).read())


class MertonEnv(gym.Env):

    def __init__(self, Mu = 0.02, Sigma = 0.15, Row = None, Risk_Aversion = 1, Rf = 0.02, Min_Leverage = -5, Max_Leverage = 5, Time_Horizon = 50, Time_Step = 1/252, Model = "GBM", Intermediate_Reward = False):

        if isinstance(Mu, int) or isinstance(Mu, float):
            self.RETURN_FUNC = self.Single_Asset_GBM
            self.action_space = spaces.Box(low = Min_Leverage, high = Max_Leverage, shape = (1,))
        else:
            self.RETURN_FUNC = self.Multi_Asset_GBM
            self.action_space = spaces.Box(low = np.array([Min_Leverage] * Mu.size), high = np.array([Max_Leverage] * Mu.size))

        self.MU            = Mu
        self.SIGMA         = Sigma
        self.ROW           = Row
        self.RISK_AVERSION = Risk_Aversion
        self.RF            = Rf
        self.MIN_LEVERAGE  = Min_Leverage
        self.MAX_LEVERAGE  = Max_Leverage
        self.TIME_HORIZON  = Time_Horizon
        self.TIME_STEP     = Time_Step
        self.INTERMEDIATE_REWARD = Intermediate_Reward
        # if MODEL == SOME_OTHER_MODEL ...

        self.Wealth = np.random.uniform()
        self.Tau    = self.TIME_HORIZON
        self.State  = np.array([self.Tau, self.Wealth])

        self.Done   = False
        self.Reward = 0

        # 0: Tau
        # 1: Wealth
        self.observation_space = spaces.Box(low = np.array([0.0, 0.0]), high = np.array([self.TIME_HORIZON, 100]))


    def step(self, action):
        assert self.action_space.contains(action), "Action %r (of type %s) is not within the action space" % (action, type(action))

        if self.Done == True:
            warnings.warn("Action attempted when episode is over.", Warning)

        if self.RISK_AVERSION == 1 and self.INTERMEDIATE_REWARD == True:
            self.Reward = self.Utility()

        self.Wealth *= self.RETURN_FUNC(action)
        self.Tau    -= self.TIME_STEP
        self.State   = np.array([self.Tau, self.Wealth])

        # Old Version
        # Period_Return  = self.RETURN_FUNC(self.MU, self.SIGMA, self.TIME_STEP, self.ROW) - (self.RF * self.TIME_STEP)
        # self.Wealth   *= (1 + (self.RF * self.TIME_STEP) + (a * Period_Return))


        if self.RISK_AVERSION == 1 and self.INTERMEDIATE_REWARD == True:
            self.Reward = (self.Utility() - self.Reward)
        else:
            self.Reward = 0

        self.Done  = False
        if self.Tau <= 0 or self.Wealth <= 0:
            self.Done = True
            self.Reward = self.Utility()
            self.State[0] = 0

        return self.State, self.Reward, self.Done, {}


    def reset(self):
        # Must return an intial Observation
        self.Wealth = np.random.uniform()
        self.Tau    = self.TIME_HORIZON
        self.State  = np.array([self.Tau, self.Wealth])

        self.Done = False
        self.Reward = 0

        return self.State


    def render(self):
        print("Current Wealth: " + str(round(self.Wealth, 4)) + ", Tau: " + str(self.Tau))





    def Single_Asset_GBM(self, Action):
        # Generate returns from a normal distribution
        Mean = (self.MU - (self.SIGMA ** 2) / 2) * self.TIME_STEP
        Std = self.SIGMA * (self.TIME_STEP ** 0.5)
        Return = (np.exp(np.random.normal(Mean, Std)) - 1) - (self.RF * self.TIME_STEP)
        Net_Return = (1 + (self.RF * self.TIME_STEP) + (Action[0] * Return))

        return Net_Return


    def Multi_Asset_GBM(self, Action):
        # Genrate the means and standard deviations
        Means = (self.MU - (self.SIGMA ** 2) / 2) * self.TIME_STEP
        Stds  = (self.SIGMA * (self.TIME_STEP ** 0.5)).reshape(-1,1)
        Covs  = self.ROW * np.matmul(Stds, Stds.T)

        Returns = (np.exp(np.random.multivariate_normal(Means, Covs)) - 1) - (self.RF * self.TIME_STEP)
        Net_Return = (1 + (self.RF * self.TIME_STEP) + np.sum(Action * Returns))

        return Net_Return


    def Utility(self):
        if self.Wealth <= 0:
            return -10

        if self.RISK_AVERSION == 1:
            return np.log(self.Wealth)

        else:
            return (self.Wealth ** (1 - self.RISK_AVERSION)) / (1 - self.RISK_AVERSION)























# End.
