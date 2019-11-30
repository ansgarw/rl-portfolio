import numpy  as np
import pandas as pd
import warnings
import gym
import os
from scipy.optimize import minimize

# Global "Constants"
_Default_Filename = os.path.realpath(__file__).replace("PortfolioEnvs.py", "Monthly_Data.csv")

class Portfolio_Env(gym.Env):

    '''
    A gym environment used to train RL Portfolio management agents with historical data.
    The environment comes with a default dataset, or another may be specified by the user.
    '''

    def __init__ (self, **kwargs):
        '''
        Initialise the class.
        Since user arguments can no longer be passed to gym.make, Set_Params must be called subsequently to overwrite the default arguments.
        '''

        self.Time_Step           = kwargs['Time_Step']
        self.Risk_Aversion       = kwargs['Risk_Aversion']
        self.Episode_Length      = kwargs['Episode_Length']
        self.Max_Leverage        = kwargs['Max_Leverage']
        self.Min_Leverage        = kwargs['Min_Leverage']
        self.Validation_Frac     = kwargs['Validation_Frac']
        self.Intermediate_Reward = kwargs['Intermediate_Reward']

        self.is_Normalised = False
        self.Multi_Asset_Method = 'Equal_Weight'

        # Flag used to distinguish a training episode from a validation epiodse, to facilitate cross val.
        self.isTraining = True

        self.Wealth = np.clip(np.random.normal(1, 0.25), 0.25, 1.75)
        self.action_space = gym.spaces.Box(low = kwargs['Min_Leverage'], high = kwargs['Max_Leverage'], shape = (1,), dtype = np.float32)

        # Set up the default dataset.
        Data = pd.read_csv(_Default_Filename)
        self.State_Parameters = ['DY', 'EY', 'DP', 'DE', 'svar', 'infl', 'AAA', 'BAA', 'lty', 'defaultspread', 'tbl',
                                 'b/m', 'ntis', 'ltr', 'corpr', 'CRSP_SPvw', 'CRSP_SPvwx', 'Mom', 'HML', 'SMB']

        Data = Data[self.State_Parameters + ['Fama Mkt Excess', '1M TBill']]
        Data = Data.dropna()
        Data.reset_index(drop = True, inplace = True)

        self.Return_Data = Data['Fama Mkt Excess'].values.reshape(-1,1)
        self.State_Data  = Data[self.State_Parameters].values.reshape(-1, len(self.State_Parameters))
        self.Rf          = Data['1M TBill'].values.reshape(-1,1)

        self.Accepted_Keywords = {'Risk_Aversion', 'Episode_Length', 'Max_Leverage', 'Min_Leverage', 'Validation_Frac', 'Intermediate_Reward', 'DataBase', 'State_Parameters', 'Return_Key', 'Risk_Free_Key', 'Time_Step', 'First_Difference_Params', 'Normalise', 'Multi_Asset_Method'}


        # Plus two as wealth and tau are also part of the state space.
        self.observation_space = gym.spaces.Box(low = np.array([-np.inf] * (self.State_Data.shape[1] + 2)), high = np.array([np.inf] * (self.State_Data.shape[1] + 2)), dtype = np.float32)
        self.action_space = gym.spaces.Box(low = np.array([self.Min_Leverage]), high = np.array([self.Max_Leverage]), dtype = np.float32)
        self.Reset_Validation_Data()


    def Set_Params(self, **kwargs):
        '''
        Unfortunately it is no longer possible to send arguments to gym.make (as of 19/9/19), so this method may be used to
        change the Parameters of the environment from their defaults.

        Parameters
        ----------


            Risk_Aversion | float
                The investors risk aversion, which is used to generate utility

            Validation_Frac | float
                The fraction of the data to be kept aside for out of sample validation (Default 0.3)

            Min_Leverage | float
                The smallest leverage which the agent may take. Must be negative to allow the agent too short.

            Max_Leverage | float
                The highest leverage that the agent may take

            Episode_Length | int
                The number of steps that an episode should consist of. (Time horizon and step are no longer used as period is a property of the underlying VAR model.)

            Intermediate_Rewards | bool
                A flag indicating whether the environment will return intermediate rewards. Note for this parameter to have an effect Risk_Aversion must equal one, or it will default to false. Intermediate_Reward are calculated as the increase in utility across the step.

            Time_Step | float
                The length of time that one step through the database corresponds to. Cannot be overwritten unless a custom database is being used.



            State_Parameters | list
                A list of the parameters to be used to construct the state. (list of parameters availiable in the default database below)

            Return_Key | string, list
                The key of the column to be used as returns. If a list is passed then the environment will run with multiple assets. (list of parameters availiable in the default database below)

            Risk_Free_Key | string
                The key of the column to be used as the risk free return. (list of parameters availiable in the default database below)

            DataBase | pandas.DataFrame
                A pandas dataframe containing a custom database if desired. If this is overloaded, State_Parameters, Return_Key and Risk_Free_Key must also be specified.


            Multi_Asset_Method | string
                The benchmark to use for multi-asset investing. Either 'Equal_Weight' or 'Treynor_Black'


            First_Difference_Params | list
                A list of parameters whose first difference rather than absolute value should be used as a state paremeter


            Normalise | bool
                A boolean indicating if the State data should be normalised. Default = False



        Default Dataset Keys
        --------------------

        Excess Return
        -------------
            Fama Mkt Excess  - Excess market return by Fama
            Return           - Excess return on the S&P500 Index


        Risk Free Rate
        --------------
            Rfree      - Risk free rate (source unknown, from original predictorData)
            1M TBill   - Fama french one month T Bill return.


        State_Parameters
        ----------------
            Below is a list of the available state parameters in the Monthly-v1 version, and the date from which they
            become available. NaNs are dropped automatically, however pay attention to starting date to ensure the
            amount of available data is not reduced too sevearly.

            1871-01 : ['D12', 'E12', 'Rfree', 'DY', 'EY', 'DP', 'DE']
            1885-02 : ['svar']
            1920-02 : ['infl', 'AAA', 'BAA', 'lty', 'defaultspread', 'tbl']
            1926-07 : ['SMB', 'HML']
            1927-01 : ['b/m', 'ntis', 'ltr', 'corpr', 'CRSP_SPvw', 'CRSP_SPvwx', 'Mom']
            1930-12 : ['BAB']
            1937-05 : ['csp']
            1947-01 : ['CPIAUCSL']
            1982-01 : ['Term spread']

            Data ends 2019.


        Notes
        -----
            Parameters ['D12', 'E12'] are non-stationary, hence must not be included in the state parameters
            Parameter 'DE' contains a single (anomolous) entry which is 4x higher than average.

        '''

        if 'DataBase' in kwargs.keys():
            assert set(['Return_Key', 'Risk_Free_Key', 'State_Parameters']).issubset(set(kwargs.keys())), 'Please specify State_Parameters, Return_Key and Risk_Free_Key when using a custom database.'

        if 'Time_Step' in kwargs.keys():
            assert 'DataBase' in kwargs.keys(), 'Only specify Time_Step if a new DataBase is in use.'

        if len(set(kwargs.keys()).intersection({'DataBase', 'Return_Key', 'Risk_Free_Key', 'State_Parameters', 'First_Difference_Params'})) > 0:

            if 'Normalise'           in kwargs.keys() : self.is_Normalised       = kwargs['Normalise']
            if 'Time_Step'           in kwargs.keys() : self.Time_Step           = kwargs['Time_Step']
            if 'Max_Leverage'        in kwargs.keys() : self.Max_Leverage        = kwargs['Max_Leverage']
            if 'Min_Leverage'        in kwargs.keys() : self.Min_Leverage        = kwargs['Min_Leverage']
            if 'Risk_Aversion'       in kwargs.keys() : self.Risk_Aversion       = kwargs['Risk_Aversion']
            if 'Episode_Length'      in kwargs.keys() : self.Episode_Length      = kwargs['Episode_Length']
            if 'Validation_Frac'     in kwargs.keys() : self.Validation_Frac     = kwargs['Validation_Frac']
            if 'State_Parameters'    in kwargs.keys() : self.State_Parameters    = kwargs['State_Parameters']
            if 'Multi_Asset_Method'  in kwargs.keys() : self.Multi_Asset_Method  = kwargs['Multi_Asset_Method']
            if 'Intermediate_Reward' in kwargs.keys() : self.Intermediate_Reward = kwargs['Intermediate_Reward']

            Data          = kwargs['DataBase']      if 'DataBase'      in kwargs.keys() else pd.read_csv(_Default_Filename)
            Return_Key    = kwargs['Return_Key']    if 'Return_Key'    in kwargs.keys() else 'Fama Mkt Excess'
            Risk_Free_Key = kwargs['Risk_Free_Key'] if 'Risk_Free_Key' in kwargs.keys() else '1M TBill'

            assert Return_Key != Risk_Free_Key, 'Return key may not be the same as risk free key.'

            if isinstance(Return_Key, list):
                Keys = set(self.State_Parameters)
                Keys.update(Return_Key + [Risk_Free_Key])
                Data = Data[list(Keys)]
            else:
                Keys = set(self.State_Parameters)
                Keys.update([Return_Key, Risk_Free_Key])
                Data = Data[list(Keys)]


            Data = Data.dropna()
            Data.reset_index(drop = True, inplace = True)

            if 'First_Difference_Params' in kwargs.keys():
                for parameter in kwargs['First_Difference_Params']:
                    Data[parameter] = Data[parameter].pct_change()

                Data = Data.iloc[1:]
                # print(Data[Data.isna().any(axis=1)])

                for parameter in kwargs['First_Difference_Params']:
                    assert np.isfinite(np.all(Data[parameter].values)), 'Factor ' + parameter + ' contains inf after taking first difference.'

            self.Return_Data = Data[Return_Key].values if isinstance(Return_Key, list) else Data[Return_Key].values.reshape(-1,1)
            self.State_Data  = Data[self.State_Parameters].values
            self.Rf          = Data[Risk_Free_Key].values.reshape(-1,1)

        else:
            if 'Normalise'           in kwargs.keys() : self.is_Normalised       = kwargs['Normalise']
            if 'Max_Leverage'        in kwargs.keys() : self.Max_Leverage        = kwargs['Max_Leverage']
            if 'Min_Leverage'        in kwargs.keys() : self.Min_Leverage        = kwargs['Min_Leverage']
            if 'Risk_Aversion'       in kwargs.keys() : self.Risk_Aversion       = kwargs['Risk_Aversion']
            if 'Episode_Length'      in kwargs.keys() : self.Episode_Length      = kwargs['Episode_Length']
            if 'Validation_Frac'     in kwargs.keys() : self.Validation_Frac     = kwargs['Validation_Frac']
            if 'Multi_Asset_Method'  in kwargs.keys() : self.Multi_Asset_Method  = kwargs['Multi_Asset_Method']
            if 'Intermediate_Reward' in kwargs.keys() : self.Intermediate_Reward = kwargs['Intermediate_Reward']


        self.action_space = gym.spaces.Box(low = np.array([self.Min_Leverage] * self.Return_Data.shape[1]), high = np.array([self.Max_Leverage] * self.Return_Data.shape[1]), dtype = np.float32)
        self.observation_space = gym.spaces.Box(low = np.array([-np.inf] * (len(self.State_Parameters) + 2)), high = np.array([np.inf] * (len(self.State_Parameters) + 2)), dtype = np.float32)

        for key in kwargs.keys():
            if not key in self.Accepted_Keywords:
                print("Keyword:", key, "not recognised.")

        if self.Risk_Aversion != 1 and self.Intermediate_Reward == True:
            print("Warning: Intermediate Reward as no effect when Risk Aversion is not equal to one.")

        self.Reset_Validation_Data()


    def Reset_Validation_Data (self, Set_last = True):
        '''
        Reset the location of the holdout validation dataset (In order to prevent overfitting hyper parameters to the validation set.)

        Parameters
        ----------
            Set_last | bool
                Indicates whether the validation set should be the last x percent of the dataset.
        '''

        if Set_last == False:
            self.Validation_Start = np.random.randint(low = self.Episode_Length, high = (self.Return_Data.shape[0] * (1 - self.Validation_Frac)) - self.Episode_Length)
            self.Validation_End   = self.Validation_Start + int(self.Return_Data.shape[0] * self.Validation_Frac)
        else:
            self.Validation_Start = int(self.Return_Data.shape[0] * (1 - self.Validation_Frac))
            self.Validation_End   = int(self.Return_Data.shape[0])

        if self.is_Normalised == True:
            inSample_Mean = np.mean(np.vstack((self.State_Data[0:self.Validation_Start], self.State_Data[self.Validation_End:])), axis = 0)
            inSample_Std = np.std(np.vstack((self.State_Data[0:self.Validation_Start], self.State_Data[self.Validation_End:])), axis = 0)
            self.State_Data = (self.State_Data - inSample_Mean) / inSample_Std

        Mean = np.mean(np.vstack((self.Return_Data[0:self.Validation_Start], self.Return_Data[self.Validation_End:])), axis = 0)
        Vars = np.var(np.vstack((self.Return_Data[0:self.Validation_Start], self.Return_Data[self.Validation_End:])), axis = 0)

        self.Training_Var    = Vars
        self.Training_Mean   = Mean
        self.Training_Merton = self.Merton_Fraction()


    def Over_Sample (self, Mult, N_ = 5, P_ = 2):

        '''
        Parameters
        ----------
            Mult | float
                The number of synthetic observations to generate, as a multiple of the number of observations in the original dataset.

            N_ | int
                The number of neighbours to find, and randomly select between for interpolation

            P_ | float
                Coefficient for Minkowski distance


        Notes
        -----
            1. Synthetic observations will exhibit no autocorrolation, and hence the problem becomes a fully observed MDP, and using a RNN based AC makes no sense.
            2. The Minkowski distance is used to determine the nearest neighbours (defualt p = 2, giving euclidean distance)
            3. Synthetic observations are appended to the end of the training dataset, so the time-series data is unaffected.
            4. Persuant to the original paper proposing SMOTE, synthetics will not be generated from other synthetics.

        '''

        # Step one - Normalise the training data.
        if self.is_Normalised == True:
            Original_State = np.vstack((self.State_Data[:self.Validation_Start], self.State_Data[self.Validation_End:]))
        else:
            inSample_Mean = np.mean(np.vstack((self.State_Data[0:self.Validation_Start], self.State_Data[self.Validation_End:])), axis = 0)
            inSample_Std = np.std(np.vstack((self.State_Data[0:self.Validation_Start], self.State_Data[self.Validation_End:])), axis = 0)
            Original_State = (np.vstack((self.State_Data[:self.Validation_Start], self.State_Data[self.Validation_End:])) - inSample_Mean) / inSample_Std

        Original_Return = np.vstack((self.Return_Data[:self.Validation_Start], self.Return_Data[self.Validation_End:]))
        Original_Rf = np.vstack((self.Rf[:self.Validation_Start], self.Rf[self.Validation_End:]))


        # Step Two - Generate synthetic observations.
        # 1. Pick a datapoint at random
        # 2. Find its N nearest neighbours
        # 3. Select one of thgem at random
        # 4. Interpolate between the two points to genetate a synthetic observation.

        def Nearest_Neighbours (Observation, N, P):
            '''
            Returns the index of the N nearest neighbours of the Observation containied within the variable 'Original_State'.

            Parameters
            ----------
                N | int
                    The number of neighbours to find.

                Observation | np.array (1D)
                    The observation whose neighbours are to be found.

            '''

            Distance = np.sum((Original_State - Observation) ** P, axis = 1) ** (1/P)
            return np.argsort(Distance)[1 : N+1]

        Synthetic_Obs = np.zeros((Mult * Original_State.shape[0], Original_State.shape[1]))
        Synthetic_Returns = np.zeros((Mult * Original_Return.shape[0], Original_Return.shape[1]))
        Synthetic_Rf = np.zeros((Mult * Original_Rf.shape[0], Original_Rf.shape[1]))

        for i in range(Synthetic_Obs.shape[0]):
            Obs_Index = np.random.randint(low = 0, high = Original_State.shape[0])
            Neighbour_Index = np.random.choice(Nearest_Neighbours(Original_State[Obs_Index], N_, P_), 1)

            Interpolate_Frac = np.random.uniform()
            Synthetic_Obs[i] = (Original_State[Neighbour_Index] - Original_State[Obs_Index]) * Interpolate_Frac + Original_State[Obs_Index]
            Synthetic_Returns[i] = (Original_Return[Neighbour_Index] - Original_Return[Obs_Index]) * Interpolate_Frac + Original_Return[Obs_Index]
            Synthetic_Rf[i] = (Original_Rf[Neighbour_Index] - Original_Rf[Obs_Index]) * Interpolate_Frac + Original_Rf[Obs_Index]

        if self.is_Normalised == False:
            # Un-normalise the data
            Synthetic_Obs = (Synthetic_Obs + inSample_Mean) * inSample_Std

        self.State_Data = np.vstack((self.State_Data, Synthetic_Obs))
        self.Return_Data = np.vstack((self.Return_Data, Synthetic_Returns))
        self.Rf = np.vstack((self.Rf, Synthetic_Rf))


    def reset (self):

        '''
        Resets the environment so a new episode may be ran. Must be called by the user / agent to begin an episode.

        Returns
        -------
            Oberservation | np.array (1D)
                The oberservation space of this environment will include [Wealth, Tau] as well as all of the paramters specified in State_Parameters

        Notes
        -----
            Wealth is initalised as a uniform random variable about 1 as that is the range across which the utlity curve's gradient variaes the most, and a random starting wealth helps the agents to experiance many wealths, and hence better map the value function. When validating however wealth is always instanced at 1, for consistency.

            Since this environment is based in historical data, reset must select at random a new starting Index for the next episode.
        '''

        if self.isTraining == True:
            Possible_Indexes = np.append(np.arange(self.Validation_Start - self.Episode_Length), np.arange(self.Validation_End, self.Return_Data.shape[0] - self.Episode_Length))
            self.Start_Index = np.random.choice(Possible_Indexes)
            self.Wealth = np.clip(np.random.normal(1, 0.25), 0.25, 1.75)
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
        Steps the environment forward, this is the main point of interface between an agent and the environment.

        Parameters
        ----------
            Action | np.array (1D)
                The array must have the same length as the number of assets in the environment. The Nth value in the array represents the fraction of ones wealth to invest in the Nth asset. Negative values may be sent to enter short positions, or values with abs() > 1 to leverage oneself.

        Returns
        -------
            A tuple of the following 4 things:

            1. Observation | np.array (1D)
                The oberservation space of this environment will include [Wealth, Tau] as well as all of the paramters specified in State_Parameters.

            2. Reward | float
                The reward for the last action taken.

            3. Done | bool
                Indicates whether the current episode has ended or not. Once an episode has ended the envrionment must be reset before another step may be taken.

            4. Info | dict
                A dictionary which includes extra information about the state of the environment which is not included in the observation as it is not relevent to the agent. Currently this includes:
                    'Mkt-Rf' : The excess return of the market on the last step
                    'Rfree'  : The risk free rate.
        '''

        assert hasattr(self, 'Done'), 'Environment must be reset before the episode starts'
        assert self.Done == False, "Attempt to take an action whilst Done == True"
        assert self.action_space.contains(Action), "Action %r is not within the action space" % (Action)

        self.Index += 1

        Return = (1 + self.Rf[self.Index][0] + np.sum(self.Return_Data[self.Index] * Action))

        if self.Intermediate_Reward == True and self.Risk_Aversion == 1:
            self.Reward = self.Utility(self.Wealth * Return) - self.Utility()
        else:
            self.Reward = 0

        self.Wealth *= Return

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
        '''
        Generates an observation (For internal use only)

        Returns
        -------
            np.array (1D)
                An observation
        '''

        Tau   = (self.End_Index - self.Index) / (1 / self.Time_Step)
        State = self.State_Data[self.Index] if len(self.State_Parameters) > 0 else []
        return np.append([self.Wealth, Tau], State)


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

        Info = {'Mkt-Rf' : self.Return_Data[self.Index],
                'Rfree'  : self.Rf[self.Index][0]}

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
        Calculates the Merton portfolio fraction if there is only one asset, or the Equal_Weight / Treynor_Black portfolio and Merton portfolio fraction if multiple assets are being used.

        Returns
        -------
            float
                In the case of a single asset. Represents the Merton Portfolio Fraction

            np.array (1D)
                In the case of multiple assets. Represents the optimal fraction of ones wealth to invest in each asset.

        Notes
        -----

        '''

        if self.Return_Data.shape[1] == 1:
            return self.Training_Mean[0] / ((self.Training_Var[0]) * self.Risk_Aversion)

        else:

            # Since Markowitz is not robust for many assets (especially if they have high corrolation), here we use a Treynor Black portfolio as a benchmark
            # The index for calculating this method is defined as an equal weighted portfolio of the N assets

            if self.Multi_Asset_Method == 'Equal_Weight':

                Index_Weights = np.ones(self.Return_Data.shape[1]) / self.Return_Data.shape[1]
                Training_Returns = np.vstack((self.Return_Data[0:self.Validation_Start], self.Return_Data[self.Validation_End:]))
                Index_Returns = np.matmul(Training_Returns, Index_Weights.reshape(-1,1))

                Merton_Leverage = np.mean(Index_Returns) / (self.Risk_Aversion * np.var(Index_Returns))

                return Index_Weights * Merton_Leverage

            else :

                Index_Weights = np.ones(self.Return_Data.shape[1]) / self.Return_Data.shape[1]
                Training_Returns = np.vstack((self.Return_Data[0:self.Validation_Start], self.Return_Data[self.Validation_End:]))
                Index_Returns = np.matmul(Training_Returns, Index_Weights.reshape(-1,1))

                Index_Var = np.var(Index_Returns)
                Index_Mean = np.mean(Index_Returns)

                # Now calculate the beta for each asset
                Betas  = np.zeros(self.Return_Data.shape[1])
                Alphas = np.zeros(self.Return_Data.shape[1])
                Idiosyncratic_Var = np.zeros(self.Return_Data.shape[1])

                for i in range(self.Return_Data.shape[1]):
                    Betas[i] = np.cov(np.hstack((Training_Returns[:,i].reshape(-1,1), Index_Returns.reshape(-1,1))))[0,1] / Index_Var
                    Alphas[i] = np.mean(Training_Returns[:,i]) - Index_Mean * Betas[i]
                    Idiosyncratic_Var[i] = np.var(Training_Returns[:,i]) - Index_Var * Betas[i]


                # Now we can calculate portfolio weights, overweighting stocks with positive alpha relative to their Idiosyncratic_Var
                Active_Weights = Alphas / Idiosyncratic_Var
                Passive_Weights = ((Index_Mean / Index_Var) - np.sum(Betas * Active_Weights)) / self.Return_Data.shape[0]

                Portfolio_Weights = Active_Weights + Passive_Weights
                Portfolio_Returns = np.matmul(Training_Returns, Portfolio_Weights)

                Merton_Leverage = np.mean(Portfolio_Returns) / (self.Risk_Aversion * np.var(Portfolio_Returns))
                return (Portfolio_Weights * Merton_Leverage).flatten()


    def Validate (self, N_Episodes, Agent):
        '''
        A validation function used to appraise the performance of an agent across N episodes.

        Parameters
        ----------
            N_Episodes | int
                Deprecated

            Agent | A compatible AC or DQN Agent.
                The Agent to validate.

        Returns
        -------

        '''

        Merton_Results = {'Mean_Utility' : [],
                          'Mean_Return'  : [],
                          'Return_Std'   : [],
                          'Sharpe'       : []}
        Agent_Results  = {'Mean_Utility' : [],
                          'Mean_Return'  : [],
                          'Return_Std'   : [],
                          'Sharpe'       : []}


        self.isTraining = False

        self.Start_Index = self.Validation_Start
        self.End_Index = self.Validation_End - 1
        self.Index = self.Start_Index

        self.Wealth = 1
        self.Done = False
        Ep_Utility = np.ones(2) # 0 : Merton, 1 : Agent

        State_0 = self.Gen_State()
        State_0[1] = self.Episode_Length * self.Time_Step * 0.9

        i = 0
        while self.Done == False:
            Action = Agent.Predict_Action(State_0, OOS = True)
            State_1, Reward, Done, Info = self.step(Action.flatten())
            State_1[1] = self.Episode_Length * self.Time_Step * 0.9
            State_0 = State_1

            i += 1

            Merton_Results['Mean_Return'].append(Info['Rfree'] + np.sum(Info['Mkt-Rf'] * self.Training_Merton))
            Agent_Results['Mean_Return'].append(Info['Rfree'] + np.sum(Info['Mkt-Rf'] * Action))
            Merton_Results['Sharpe'].append(np.sum(Info['Mkt-Rf'] * self.Training_Merton))
            Agent_Results['Sharpe'].append(np.sum(Info['Mkt-Rf'] * Action))
            Ep_Utility[0] *= 1 + Info['Rfree'] + np.sum(Info['Mkt-Rf'] * self.Training_Merton)
            Ep_Utility[1] *= 1 + Info['Rfree'] + np.sum(Info['Mkt-Rf'] * Action)

            if i % self.Episode_Length == 0:
                Merton_Results['Mean_Utility'].append(Ep_Utility[0])
                Agent_Results['Mean_Utility'].append(Ep_Utility[1])
                Ep_Utility = np.ones(2)

        Merton_Results['Mean_Utility'] = np.mean(self.Utility(np.array(Merton_Results['Mean_Utility'])))
        Merton_Results['Return_Std']   = np.std(Merton_Results['Mean_Return']) * ((1 / self.Time_Step) ** 0.5)
        Merton_Results['Mean_Return']  = np.mean(Merton_Results['Mean_Return']) / self.Time_Step
        Merton_Results['Sharpe']       = (np.mean(Merton_Results['Sharpe']) / self.Time_Step) / Merton_Results['Return_Std']

        Agent_Results['Mean_Utility'] = np.mean(self.Utility(np.array(Agent_Results['Mean_Utility'])))
        Agent_Results['Return_Std']   = np.std(Agent_Results['Mean_Return']) * ((1 / self.Time_Step) ** 0.5)
        Agent_Results['Mean_Return']  = np.mean(Agent_Results['Mean_Return']) / self.Time_Step
        Agent_Results['Sharpe']       = (np.mean(Agent_Results['Sharpe']) / self.Time_Step) / Agent_Results['Return_Std']

        self.isTraining = True
        return Merton_Results, Agent_Results


    def Equity_Curve (self, Agent):
        '''
        Draws an equity curve representing the return on investment of holding the Merton Portfolio Fraction, and the Agent Portfolio.

        Notes
        -----
            At some point we may remove tau from the state space, or make its includion optimal.
            If that happens this function must be adjusted, at the moment tau is kept at Episode_Length * Time_Step * 0.9, as any
            higher figure will cause issues.
        '''

        self.isTraining = False

        self.Start_Index = self.Validation_Start
        self.End_Index = self.Validation_End - 1
        self.Index = self.Start_Index

        self.Wealth = 1
        self.Done = False

        Agent_Equity = [1]
        Merton_Equity = [1]

        State_0 = self.Gen_State()
        State_0[1] = self.Episode_Length * self.Time_Step * 0.9

        while self.Done == False:
            Action = Agent.Predict_Action(State_0, OOS = True)
            State_1, Reward, Done, Info = self.step(Action.flatten())
            State_1[1] = self.Episode_Length * self.Time_Step * 0.9
            State_0 = State_1

            Merton_Equity.append(Merton_Equity[-1] * (1 + Info['Rfree'] + np.sum(Info['Mkt-Rf'] * self.Training_Merton)))
            Agent_Equity.append(State_0[0])

        return {'Agent' : Agent_Equity, 'Merton' : Merton_Equity}


class Sim_GBM_Env (Portfolio_Env):

    def __init__ (self, **kwargs):

        super(Sim_GBM_Env, self).__init__(**kwargs)

        self.Mu    = kwargs['Mu']
        self.Rf_   = kwargs['Rf']
        self.Row   = kwargs['Row']
        self.Sigma = kwargs['Sigma']

        self.DataBase_Len = 200000

        self.Accepted_Keywords = self.Accepted_Keywords.union({'Mu', 'Sigma', 'Row', 'Rf', 'DataBase_Len'})


        self.Setup()


    def Set_Params (self, **kwargs):

        if 'Mu'    in kwargs.keys() : self.Mu    = kwargs['Mu']
        if 'Rf'    in kwargs.keys() : self.Rf_   = kwargs['Rf']
        if 'Row'   in kwargs.keys() : self.Row   = kwargs['Row']
        if 'Sigma' in kwargs.keys() : self.Sigma = kwargs['Sigma']

        if 'Time_Step'    in kwargs.keys() : self.Time_Step    = kwargs['Time_Step']
        if 'DataBase_Len' in kwargs.keys() : self.DataBase_Len = kwargs['DataBase_Len']

        self.Setup(**kwargs)


    def Setup (self, **kwargs):

        if isinstance(self.Row, float):
            self.Row = np.array([[1, self.Row], [self.Row, 1]])

        if isinstance(self.Mu, int) or isinstance(self.Mu, float):
            Mean = (self.Mu - (self.Sigma ** 2) / 2) * self.Time_Step
            Std = self.Sigma * (self.Time_Step ** 0.5)
            Mkt_Return = (np.exp(np.random.normal(Mean, Std, size = self.DataBase_Len)) - 1) - (self.Rf_ * self.Time_Step)

        else:
            Means = (self.Mu - (self.Sigma ** 2) / 2) * self.Time_Step
            Stds  = (self.Sigma * (self.Time_Step ** 0.5)).reshape(-1,1)
            Covs  = self.Row * np.matmul(Stds, Stds.T)
            Mkt_Return = (np.exp(np.random.multivariate_normal(Means, Covs, size = self.DataBase_Len)) - 1) - (self.Rf_ * self.Time_Step)

        Data = pd.DataFrame(columns = ['Mkt-Rf'], data = Mkt_Return)
        Data.loc[:, 'Rf'] = np.ones(self.DataBase_Len) * self.Rf_

        kwargs['DataBase'] = Data
        kwargs['Return_Key'] = 'Mkt-Rf'
        kwargs['Risk_Free_Key'] = 'Rf'
        kwargs['State_Parameters'] = []

        super(Sim_GBM_Env, self).Set_Params(**kwargs)



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

        for i in range(N):
            Returns[i] = self.step().flatten()
            Factors[i] = self.Factor_Values[-State_Hist_Len:][::-1].flatten()

        self.reset()

        return Returns, Factors



class Sim_VAR_Env (Portfolio_Env):

    def __init__ (self, **kwargs):
        super(Sim_VAR_Env, self).__init__(**kwargs)

        self.VAR_Model = VAR_Engine(kwargs['Factor_Beta'], kwargs['Asset_Beta'], kwargs['Cov'], kwargs['Time_Step'])
        self.Rf_ = kwargs['Rf']
        self.DataBase_Len = 10000
        self.Factor_State_Len = kwargs['Factor_State_Len']

        self.Accepted_Keywords = self.Accepted_Keywords.union({'Factor_Beta', 'Asset_Beta', 'Cov', 'Factor_State_Len', 'Rf', 'DataBase_Len'})

        self.Setup(**kwargs)


    def Set_Params (self, **kwargs):

        if 'Rf'               in kwargs.keys() : self.Rf_              = kwargs['Rf']
        if 'DataBase_Len'     in kwargs.keys() : self.DataBase_Len     = kwargs['DataBase_Len']
        if 'Factor_State_Len' in kwargs.keys() : self.Factor_State_Len = kwargs['Factor_State_Len']

        if set(['Factor_Beta', 'Asset_Beta', 'Cov', 'Time_Step']).issubset(set(kwargs.keys())):
            self.VAR_Model = VAR_Engine(kwargs['Factor_Beta'], kwargs['Asset_Beta'], kwargs['Cov'], kwargs['Time_Step'])

        self.Setup(**kwargs)


    def Setup (self, **kwargs):
        if set(['Factor_Beta', 'Asset_Beta', 'Cov', 'Time_Step']).issubset(set(kwargs.keys())):
            Mkt_Returns, Factors = self.VAR_Model.Genrate_Returns(self.DataBase_Len, self.Factor_State_Len)
            Rfree = np.ones((self.DataBase_Len, 1)) * self.Rf_
            columns = ['Asset_' + str(i) for i in range(self.VAR_Model.Num_Assets)] + ['Rfree'] + ['Factor_' + str(i) for i in range(self.VAR_Model.Num_Factors)]

            Data = pd.DataFrame(columns = columns, data = np.hstack((Mkt_Returns, Rfree, Factors)))

            kwargs['DataBase'] = Data
            kwargs['Return_Key'] = ['Asset_' + str(i) for i in range(self.VAR_Model.Num_Assets)]
            kwargs['Risk_Free_Key'] = 'Rfree'
            kwargs['State_Parameters'] = ['Factor_' + str(i) for i in range(self.VAR_Model.Num_Factors)]

        super(Sim_VAR_Env, self).Set_Params(**kwargs)


    def VAR_Benchmark (self, Factor_Value):
        '''
        Returns the optimal investment when trading a single asset which follows the VAR model.
        For this benchmark to be accurate the following requirements must be met:
            1. The tradeable universe must consist only of a single asset
            2. The asset must be dependent on a single factor.
            3. Risk Aversion must be equal to one.


        Parameters
        ----------
            Factor_Value | float
                The value of the factor to calculate the optimal return for.

        '''

        assert self.Risk_Aversion == 1, 'VAR_Benchmark may not be used with any Risk_Aversion other than unity.'
        assert self.VAR_Model.Num_Assets == 1, 'VAR_Benchmark may only be used when the tradeable universe consists of a single stock'
        assert self.VAR_Model.Num_Factors == 1, 'VAR_Benchmark may only be used when the number of factors equals 1.'

        return (self.VAR_Model.Asset_Beta[0,0] + (self.VAR_Model.Asset_Beta[1,0] * Factor_Value) - np.log(1 + (self.Rf_ * self.VAR_Model.Period)) + (0.5 * self.VAR_Model.Cov[0,0])) / self.VAR_Model.Cov[0,0]
