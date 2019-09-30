import numpy  as np
import pandas as pd
import warnings
import gym
import os

# Global "Constants"
_Daily_Filename      = os.path.realpath(__file__).replace("Historical.py", "PredictorDataDaily.csv")
_Monthly_Filename    = os.path.realpath(__file__).replace("Historical.py", "PredictorDataMonthly.csv")
_Monthly_Filename_V2 = os.path.realpath(__file__).replace("Historical.py", "Monthly_Data.csv")

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

        assert kwargs['Time_Step'] in ("Daily", "Monthly", "Monthly-v1"), "Episode_Length % not recognised, Please input Daily or Monthly"

        self.Time_Step       = kwargs['Time_Step']
        self.Risk_Aversion   = kwargs['Risk_Aversion']
        self.Episode_Length  = kwargs['Episode_Length']
        self.Max_Leverage    = kwargs['Max_Leverage']
        self.Min_Leverage    = kwargs['Min_Leverage']
        self.Validation_Frac = kwargs['Validation_Frac']
        self.Intermediate_Reward = False

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
            self.Data['Risk-Free Rate'] = self.Data['Risk-Free Rate'] / 100
            self.Data = self.Data.reset_index(drop = True)

            self.X = "Mkt-RF"
            self.Y = list(self.Data.columns.values)
            self.Y.remove("Unnamed: 0")
            self.Rf = "Risk-Free Rate"


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

        elif self.Time_Step == 'Monthly-v1':
            self.Data = pd.read_csv(_Monthly_Filename_V2)

            self.Y = ['D12', 'E12', 'Rfree', 'DY', 'EY', 'DP', 'DE', 'svar', 'infl', 'AAA', 'BAA', 'lty', 'defaultspread', 'tbl',
                      'b/m', 'ntis', 'ltr', 'corpr', 'CRSP_SPvw', 'CRSP_SPvwx', 'Mom']
            self.Data = self.Data[self.Y + ['Return']]
            self.Data = self.Data.dropna()
            self.Data = self.Data.reset_index(drop = True)
            self.Data['Mkt_Ret'] = self.Data['Return'] - self.Data['Rfree']
            self.X = "Mkt_Ret"
            self.Rf = "Rfree"


        # Plus two as wealth and tau are also part of the state space.
        self.observation_space = gym.spaces.Box(low = np.array([-np.inf] * (len(self.Y) + 2)), high = np.array([np.inf] * (len(self.Y) + 2)), dtype = np.float32)
        self.Reset_Validation_Data()


    def Set_Params(self, **kwargs):
        '''
        Unfortunately it is no longer possible to send arguments to gym.make (as of 19/9/19), so this method may be used to
        change the Parameters of the environment from their defaults.

        Parameters
        ----------
            Episode_Length      : The length of an episode, measured in Time_Step(s)
            Risk_Aversion       : The risk aversion of the agent
            Max_Leverage        : The maximum leverage that the bot may take without triggering an error
            Min_Leverage        : The minimum leverage that the bot may take without triggering an error
            Validation_Frac     : The holdout fraction of the dataset.
            Intermediate_Reward : A flag to indicate whether the environment should return inter-episode rewards.
            State_Parameters    : A list of the parameters to be used to construct the state.

        State_Parameters
        ----------------
            Below is a list of the available state parameters in the Monthly-v1 version, and the date from which they
            become available. NaNs are dropped automatically, however pay attention to starting date to ensure the
            amount of available data is not reduced too sevearly.

            1871-01 : ['D12', 'E12', 'Rfree', 'DY', 'EY', 'DP', 'DE']
            1885-02 : ['svar']
            1920-02 : ['infl', 'AAA', 'BAA', 'lty', 'defaultspread', 'tbl']
            1927-01 : ['b/m', 'ntis', 'ltr', 'corpr', 'CRSP_SPvw', 'CRSP_SPvwx', 'Mom']
            1930-12 : ['BAB']
            1937-05 : ['csp']
            1947-01 : ['CPIAUCSL']
            1963-07 : ['SMB', 'HML', 'RMW', 'CMA']
            1982-01 : ['Term spread']

            Data ends 2019.

        '''

        self.Risk_Aversion       = kwargs['Risk_Aversion']       if 'Risk_Aversion'       in kwargs.keys() else self.Risk_Aversion
        self.Episode_Length      = kwargs['Episode_Length']      if 'Episode_Length'      in kwargs.keys() else self.Episode_Length
        self.Max_Leverage        = kwargs['Max_Leverage']        if 'Max_Leverage'        in kwargs.keys() else self.Max_Leverage
        self.Min_Leverage        = kwargs['Min_Leverage']        if 'Min_Leverage'        in kwargs.keys() else self.Min_Leverage
        self.Validation_Frac     = kwargs['Validation_Frac']     if 'Validation_Frac'     in kwargs.keys() else self.Validation_Frac
        self.Intermediate_Reward = kwargs['Intermediate_Reward'] if 'Intermediate_Reward' in kwargs.keys() else self.Intermediate_Reward

        if self.Time_Step == 'Monthly-v1' and 'State_Parameters' in kwargs.keys():
            self.Y = kwargs['State_Parameters']

            self.Data = pd.read_csv(_Monthly_Filename_V2)
            self.Data = self.Data[self.Y + ['Return']]
            self.Data.dropna()
            self.Data = self.Data.reset_index(drop = True)
            self.Data['Return'] = self.Data['Return'] - self.Data['Rfree']


        self.action_space = gym.spaces.Box(low = self.Min_Leverage, high = self.Max_Leverage, shape = (1,), dtype = np.float32)
        self.observation_space = gym.spaces.Box(low = np.array([-np.inf] * (len(self.Y) + 2)), high = np.array([np.inf] * (len(self.Y) + 2)), dtype = np.float32)


        for key in kwargs.keys():
            if not key in ('Risk_Aversion', 'Episode_Length', 'Max_Leverage', 'Min_Leverage', 'Validation_Frac', 'Intermediate_Reward'):
                print("Keyword:", key, "not recognised.")

        if "Time_Step" in kwargs.keys():
            warnings.warn("Time_Step may not be changed, please use the daily/monthly environment")


    def Reset_Validation_Data (self, Set_last = False):
        '''
        Reset the location of the holdout validation dataset (In order to prevent overfitting hyper parameters to the validation set.)

        Parameters
        ----------
            Set_last : A boolean indicating whether the validation set should be the last x percent of the dataset.
        '''

        if Set_last == False:
            self.Validation_Start = np.random.randint(low = self.Episode_Length, high = (self.Data.shape[0] * (1 - self.Validation_Frac)) - self.Episode_Length)
            self.Validation_End   = self.Validation_Start + int(self.Data.shape[0] * self.Validation_Frac)
        else:
            self.Validation_Start = int(self.Data.shape[0] * (1 - self.Validation_Frac))
            self.Validation_End   = int(self.Data.shape[0])

        Mean = np.mean(np.append(self.Data.iloc[0:self.Validation_Start][self.X].values, self.Data.iloc[self.Validation_End::][self.X].values))
        Vars = np.var(np.append(self.Data.iloc[0:self.Validation_Start][self.X].values, self.Data.iloc[self.Validation_End::][self.X].values))

        self.Training_Merton = Mean / (Vars * self.Risk_Aversion)


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
            Possible_Indexes = np.append(np.arange(self.Validation_Start - self.Episode_Length), np.arange(self.Validation_End, self.Data.shape[0] - self.Episode_Length))
            self.Start_Index = np.random.choice(Possible_Indexes)
            self.Wealth = np.random.uniform()
        else:
            self.Start_Index = np.random.randint(low = self.Validation_Start, high = self.Validation_End - self.Episode_Length)
            # When validating we want to start each epsiode with the same wealth, as this makes the terminal utility comparable.
            self.Wealth = 1

        self.End_Index = self.Start_Index + self.Episode_Length
        self.Index = self.Start_Index

        # Ensure the training / validation split is operating properly
        if self.isTraining == True:
            assert not (self.Start_Index > self.Validation_Start and self.Start_Index < self.Validation_End), 'Validation set breached.'
            assert not (self.End_Index > self.Validation_Start and self.End_Index < self.Validation_End), 'Validation set breached.'
        else:
            assert (self.Start_Index >= self.Validation_Start and self.Start_Index <= self.Validation_End), 'Training set breached.'
            assert (self.End_Index >= self.Validation_Start and self.End_Index <= self.Validation_End), 'Training set breached.'

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

        Return = (1 + self.Data.iloc[self.Index][self.Rf] + self.Data.iloc[self.Index][self.X] * Action[0])

        if self.Intermediate_Reward == True and self.Risk_Aversion == 1:
            self.Reward = self.Utility(self.Wealth * Return) - self.Utility()
        else:
            self.Reward = 0

        self.Index += 1
        self.Wealth *= (1 + self.Data.iloc[self.Index][self.Rf] + self.Data.iloc[self.Index][self.X] * Action[0])

        if (self.Index >= self.End_Index) or (self.Wealth <= 0):
            self.Done = True
            self.Reward = self.Utility()

        return self.Gen_State(), self.Reward, self.Done, self.Gen_Info()


    def render (self):
        ''' Display some information about the current episode '''

        print("Current Wealth : " + str(self.Wealth))
        print("Starting Index : " + str(self.Start_Index))
        print("Ending Index   : " + str(self.End_Index) + "\n")


    def Gen_State (self):
        ''' Genrates a state array '''
        Tau = (self.End_Index - self.Index) / (252 if self.Time_Step == 'Daily' else 12)
        return np.append([self.Wealth, Tau], self.Data.iloc[self.Index][self.Y].values)


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

        Info = {'Mkt-Rf' : self.Data.iloc[self.Index][self.X]}

        if self.Time_Step == 'Daily':
            Info['Rfree'] = self.Data.iloc[self.Index]['Risk-Free Rate']
        elif self.Time_Step == 'Monthly' or self.Time_Step == 'Monthly-v1':
            Info['Rfree'] = self.Data.iloc[self.Index]['Rfree']

        return Info


    def Utility (self, *args):
        ''' Determine the utility of the investor at the end of life '''

        if len(args) == 1:
            Wealth = args[0]
        else:
            Wealth = self.Wealth

        if Wealth <= 0:
            return -10
        elif self.Risk_Aversion == 1:
            return np.log(Wealth)
        else:
            return (Wealth ** (1 - self.Risk_Aversion)) / (1 - self.Risk_Aversion)


    def Validate (self, N_Episodes, Agent):
        '''
        A validation function used to appraise the performance of an agent across N episodes.

        Parameters
        ----------
            N_Episodes : The number of episodes to validate across.

        Returns
        -------
            A tuple of the following:
                0. A list of terminal rewards.
                1. A list of terminal rewards holding the risk free asset
                2. A list of terminal rewards holding the ex ante merton protfolio (calculated across trianing dataset)
        '''

        Terminal_Rewards  = []
        Risk_Free_Rewards = []
        Merton_Rewards    = []

        self.isTraining = False

        for i in range(N_Episodes):
            Returns = []
            RF_Returns = []
            State = self.reset()
            Done = False

            while Done == False:
                Action = Agent.Predict_Action(State.reshape(1, self.observation_space.shape[0]))
                State, Reward, Done, Info = self.step(Action[0])
                Returns.append(Info['Mkt-Rf'])
                RF_Returns.append(Info['Rfree'])

                if Done:
                    Merton_Return = 1
                    RFree_Return  = 1
                    for i in range(len(Returns)):
                        RFree_Return  *= (1 + RF_Returns[i])
                        Merton_Return *= (1 + RF_Returns[i] + Returns[i] * self.Training_Merton)

                    Risk_Free_Rewards.append(self.Utility(RFree_Return))
                    Merton_Rewards.append(self.Utility(Merton_Return))
                    Terminal_Rewards.append(Reward)


        self.isTraining = True
        return Terminal_Rewards, Risk_Free_Rewards, Merton_Rewards
