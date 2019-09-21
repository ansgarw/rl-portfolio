import numpy  as np
import pandas as pd
import warnings
import gym
import os

# Global "Constants"
_Daily_Filename   = os.path.realpath(__file__).replace("Historical.py", "PredictorDataDaily.csv")
_Monthly_Filename = os.path.realpath(__file__).replace("Historical.py", "PredictorDataMonthly.csv")

class HistoricalEnv(gym.Env):

    def __init__ (self, **kwargs):
        '''
        Parameters
        ----------
        kwargs includes:
            Time_Step      : Options include "Daily" or "Monthly", refers to the database to use for training.
            Episode_Length : The length of an episode, measured in Time_Step(s)
            Risk_Aversion  : The risk aversion of the agent
            Max_Leverage   : The maximum leverage that the bot may take without triggering an error
            Min_Leverage   : THe minimum leverage that the bot may take without triggering an error


        Notes
        -----
            1. The environment must be reset after initialisation (by calling myEnv.reset())
        '''

        assert kwargs['Time_Step'] in ("Daily", "Monthly"), "Episode_Length % not recognised, Please input Daily or Monthly"

        self.Time_Step       = kwargs['Time_Step']
        self.Risk_Aversion   = kwargs['Risk_Aversion']
        self.Episode_Length  = kwargs['Episode_Length']
        self.Max_Leverage    = kwargs['Max_Leverage']
        self.Min_Leverage    = kwargs['Min_Leverage']
        self.Validation_Frac = kwargs['Validation_Frac']

        # Flag used to distinguish a training episode from a validation epiodse, to facilitate cross val.
        self.isTraining = True

        self.Wealth = np.random.uniform()
        self.action_space = gym.spaces.Box(low = kwargs['Min_Leverage'], high = kwargs['Max_Leverage'], shape = (1,), dtype = np.float32)


        # Read in the data...
        if self.Time_Step == "Daily":
            self.Data = pd.read_csv(_Daily_Filename)

            # "Cross-Sectional Premium" as ~7000 missing entires vs the entire rest of the database which has 270.
            self.Data.drop(columns = ["Cross-Sectional Premium"])
            self.Data = self.Data.dropna()
            self.Data["Mkt-RF"] = self.Data["Mkt-RF"] / 100
            self.Data = self.Data.reset_index(drop = True)

            self.X = "Mkt-RF"
            self.Y = list(self.Data.columns.values)
            self.Y.remove("Unnamed: 0")
            self.Rf = "RF"


        elif self.Time_Step == "Monthly":
            self.Data = pd.read_csv(_Monthly_Filename)

            # Again "csp" is missing too many entires to be included.
            self.Data.drop(columns = ["csp"])
            self.Data["Mkt_Ret"] = self.Data["Index"].pct_change() - self.Data['Rfree']
            self.Data = self.Data.dropna()
            self.Data = self.Data.reset_index(drop = True)

            self.X = "Mkt_Ret"
            self.Y = list(self.Data.columns.values)
            self.Y.remove("Unnamed: 2")
            self.Rf = "Rfree"

        # Plus two as wealth and tau are also part of the state space.
        self.observation_space = gym.spaces.Box(low = np.array([-np.inf] * (len(self.Y) + 2)), high = np.array([np.inf] * (len(self.Y) + 2)), dtype = np.float32)


    def Set_Params(self, **kwargs):
        '''
        Unfortunately it is no longer possible to send arguments to gym.make (as of 19/9/19), so this method may be used to
        change the Parameters of the environment from their defaults.

        Parameters
        ----------
            Episode_Length : The length of an episode, measured in Time_Step(s)
            Risk_Aversion  : The risk aversion of the agent
            Max_Leverage   : The maximum leverage that the bot may take without triggering an error
            Min_Leverage   : THe minimum leverage that the bot may take without triggering an error

        '''

        self.Risk_Aversion   = kwargs['Risk_Aversion']   if 'Risk_Aversion'   in kwargs.keys() else self.Risk_Aversion
        self.Episode_Length  = kwargs['Episode_Length']  if 'Episode_Length'  in kwargs.keys() else self.Episode_Length
        self.Max_Leverage    = kwargs['Max_Leverage']    if 'Max_Leverage'    in kwargs.keys() else self.Max_Leverage
        self.Min_Leverage    = kwargs['Min_Leverage']    if 'Min_Leverage'    in kwargs.keys() else self.Min_Leverage
        self.Validation_Frac = kwargs['Validation_Frac'] if 'Validation_Frac' in kwargs.keys() else self.Validation_Frac

        self.action_space = gym.spaces.Box(low = self.Min_Leverage, high = self.Max_Leverage, shape = (1,), dtype = np.float32)


        for key in kwargs.keys():
            if not key in ('Risk_Aversion', 'Episode_Length', 'Max_Leverage', 'Min_Leverage', 'Validation_Frac'):
                print("Keyword:", key, "not recognised.")

        if "Time_Step" in kwargs.keys():
            warnings.warn("Time_Step may not be changed, please use the daily/monthly environment")


    def reset (self):
        '''
        Resets the environment

        Returns : The intial state after resetting

        Notes
        -----
            1. Since this environment is based in historical data, reset must select at random a new starting Index
               for the next episode.
        '''

        if self.isTraining == True:
            self.Start_Index = np.random.randint(low = 0, high = (self.Data.shape[0] * (1 - self.Validation_Frac)) - self.Episode_Length)
            self.Wealth = np.random.uniform()
        else:
            self.Start_Index = np.random.randint(low = (self.Data.shape[0] * (1 - self.Validation_Frac)), high = self.Data.shape[0] - self.Episode_Length)
            # When validating we want to start each epsiode with the same wealth, as this makes the terminal utility comparable.
            self.Wealth = 1

        self.End_Index = self.Start_Index + self.Episode_Length
        self.Index = self.Start_Index

        self.Reward = 0
        self.Done   = False

        return self.Gen_State()


    def step (self, Action):
        '''
        Parameters
        ----------
            Action : A 1D numpy array contianing the leverage to apply to the asset.
                     Since both databases include only one asset this will be a 1D array of length 1

        Returns
        -------
            A tuple of data of the following form:
                0. State  - A 1D numpy array containing all the information required to characterise this period state
                1. Reward - The reward for the last action
                2. Done   - A boolean indicating whether the episode has ended
                3. Info   - A dictionary of additional information (currenly empty.)
        '''

        assert self.Done == False, "Attempt to take an action whilst Done == True"
        assert self.action_space.contains(Action), "Action %r is not within the action space" % (Action)

        self.Index += 1
        self.Wealth *= (1 + self.Data.iloc[self.Index][self.Rf] + self.Data.iloc[self.Index][self.X] * Action[0])

        if (self.Index >= self.End_Index) or (self.Wealth <= 0):
            self.Done = True
            self.Reward = self.Utility()

        return self.Gen_State(), self.Reward, self.Done, {}


    def render (self):
        ''' Display some information about the current episode '''

        print("Current Wealth : " + str(self.Wealth))
        print("Starting Index : " + str(self.Start_Index))
        print("Ending Index   : " + str(self.End_Index) + "\n")


    def Gen_State (self):
        ''' Genrates a state array '''
        Tau = (self.End_Index - self.Index) / (252 if self.Time_Step == 'Daily' else 12)
        return np.append([self.Wealth, Tau], self.Data.iloc[self.Index][self.Y].values)


    def Utility (self):
        ''' Determine the utility of the investor at the end of life '''
        if self.Wealth <= 0:
            return -10
        elif self.Risk_Aversion == 1:
            return np.log(self.Wealth)
        else:
            return (self.Wealth ** (1 - self.Risk_Aversion)) / (1 - self.Risk_Aversion)
