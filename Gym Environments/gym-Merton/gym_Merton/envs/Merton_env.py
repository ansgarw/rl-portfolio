import gym
from gym import error, spaces, utils
from gym.utils import seeding
import warnings
import numpy as np
import os

# CHANGELOG
# Changed max wealth in obs. space from inf to 100 fold.
# Adjusted default values for Max/Min leverage as well as Time_Step

directory = os.path.realpath(__file__).replace("Merton_env.py", "Helper.py")
exec(open(directory).read())


class MertonEnv(gym.Env):

    def __init__(self, Mu = 0.02, Sigma = 0.15, Risk_Aversion = 1, Rf = 0.02, Min_Leverage = -5, Max_Leverage = 5, Time_Horizon = 50, Time_Step = 1/252, Model = "GBM"):

        self.MU            = Mu
        self.SIGMA         = Sigma
        self.RISK_AVERSION = Risk_Aversion
        self.RF            = Rf
        self.MIN_LEVERAGE  = Min_Leverage
        self.MAX_LEVERAGE  = Max_Leverage
        self.TIME_HORIZON  = Time_Horizon
        self.TIME_STEP     = Time_Step
        self.MAX_TAU       = int(Time_Horizon / Time_Step)
        self.RETURN_FUNC   = GBM_Returns
        # if MODEL == SOME_OTHER_MODEL ...


        self.Wealth = 1.0
        self.Tau    = self.MAX_TAU
        self.State  = np.array([self.Tau, self.Wealth])

        self.Done   = False
        self.Reward = 0

        self.action_space = spaces.Box(low=self.MIN_LEVERAGE, high=self.MAX_LEVERAGE, shape=(1,), dtype=np.float32)

        # 0: Tau
        # 1: Wealth
        self.observation_space = spaces.Box(low = np.array([0.0, 0.0]), high = np.array([self.MAX_TAU, 100]))



    def step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))

        if self.Done == True:
            warnings.warn("Action attempted when episode is over.", Warning)

        a = action[0]
        Period_Return  = self.RETURN_FUNC(self.MU, self.SIGMA, self.TIME_STEP) - (self.RF * self.TIME_STEP)
        if self.RISK_AVERSION == 1:
            self.Reward    = np.log(self.Wealth)
        self.Wealth   *= (1 + (self.RF * self.TIME_STEP) + (a * Period_Return))
        self.Tau      -= 1
        self.State     = np.array([self.Tau, self.Wealth])

        if self.RISK_AVERSION == 1:
            self.Reward = (np.log(self.Wealth) - self.Reward) * 100
        else:
            self.Reward = 0

        self.Done   = False
        if self.Tau == 0:
            self.Done = True
            self.Reward = Power_Utility(self.Wealth, self.RISK_AVERSION) * 100

        return self.State, self.Reward, self.Done, {}



    def reset(self):
        # Must return an intial Observation
        self.Wealth = 1
        self.Tau    = self.MAX_TAU
        self.State  = np.array([self.Tau, self.Wealth])

        self.Done = False
        self.Reward = 0

        return self.State



    def render(self):
        print("Current Wealth: " + str(round(self.Wealth, 4)) + ", Tau: " + str(self.Tau))






















# End.
