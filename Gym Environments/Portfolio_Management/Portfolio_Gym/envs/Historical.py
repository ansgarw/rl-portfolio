import numpy  as np
import pandas as pd
import warnings
import gym
import os
from scipy.optimize import minimize


# If you mess it up just pull from the Git.
# The first objective is to remove any use of pandas in the high frequency areas of the environment

# Previous versions of the historical environment were bloated with default settings for a myriad of CSVs.
# Now the enviornment has onde Defualt csv and may be overloaded with a different one by the user.

# Global "Constants"
_Default_Filename = os.path.realpath(__file__).replace("Historical.py", "Monthly_Data.csv")

class HistoricalEnv(gym.Env):

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

        # Flag used to distinguish a training episode from a validation epiodse, to facilitate cross val.
        self.isTraining = True

        self.Wealth = np.clip(np.random.normal(1, 0.25), 0.25, 1.75)
        self.action_space = gym.spaces.Box(low = kwargs['Min_Leverage'], high = kwargs['Max_Leverage'], shape = (1,), dtype = np.float32)

        # Set up the default dataset.
        self.Data = pd.read_csv(_Default_Filename)
        self.State_Parameters = ['D12', 'E12', 'DY', 'EY', 'DP', 'DE', 'svar', 'infl', 'AAA', 'BAA', 'lty', 'defaultspread', 'tbl',
                                 'b/m', 'ntis', 'ltr', 'corpr', 'CRSP_SPvw', 'CRSP_SPvwx', 'Mom', 'HML', 'SMB']

        self.Data = self.Data[self.State_Parameters + ['Fama Mkt Excess', '1M TBill']]
        self.Data = self.Data.dropna()
        self.Data.reset_index(drop = True, inplace = True)

        self.Return_Data = self.Data['Fama Mkt Excess'].values.reshape(-1,1)
        self.State_Data  = self.Data[self.State_Parameters].values.reshape(-1, len(self.State_Parameters))
        self.Rf          = self.Data['1M TBill'].values.reshape(-1,1)


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

        '''

        self.Risk_Aversion       = kwargs['Risk_Aversion']       if 'Risk_Aversion'       in kwargs.keys() else self.Risk_Aversion
        self.Episode_Length      = kwargs['Episode_Length']      if 'Episode_Length'      in kwargs.keys() else self.Episode_Length
        self.Max_Leverage        = kwargs['Max_Leverage']        if 'Max_Leverage'        in kwargs.keys() else self.Max_Leverage
        self.Min_Leverage        = kwargs['Min_Leverage']        if 'Min_Leverage'        in kwargs.keys() else self.Min_Leverage
        self.Validation_Frac     = kwargs['Validation_Frac']     if 'Validation_Frac'     in kwargs.keys() else self.Validation_Frac
        self.Intermediate_Reward = kwargs['Intermediate_Reward'] if 'Intermediate_Reward' in kwargs.keys() else self.Intermediate_Reward

        if 'DataBase' in kwargs.keys():
            assert set(['Return_Key', 'Risk_Free_Key', 'State_Parameters']).issubset(set(kwargs.keys())), 'Please specify State_Parameters, Return_Key and Risk_Free_Key when using a custom database.'

            if 'Time_Step' in kwargs.keys() : self.Time_Step = kwargs['Time_Step']

            # Set up the new dataset.
            self.Data = kwargs['DataBase']
            self.State_Parameters = kwargs['State_Parameters']

            if isinstance(kwargs['Return_Key'], list):
                self.Data = self.Data[self.State_Parameters + kwargs['Return_Key'] + [kwargs['Risk_Free_Key']]]
                self.Data = self.Data.dropna()
                self.Data.reset_index(drop = True, inplace = True)

                self.Return_Data = self.Data[kwargs['Return_Key']].values.reshape(-1, len(kwargs['Return_Key']))
                self.State_Data  = self.Data[self.State_Parameters].values.reshape(-1, len(self.State_Parameters))
                self.Rf          = self.Data[kwargs['Risk_Free_Key']].values.reshape(-1, 1)

            else:
                self.Data = self.Data[self.State_Parameters + [kwargs['Return_Key'], kwargs['Risk_Free_Key']]]
                self.Data = self.Data.dropna()
                self.Data.reset_index(drop = True, inplace = True)

                self.Return_Data = self.Data[kwargs['Return_Key']].values.reshape(-1,1)
                self.State_Data  = self.Data[self.State_Parameters].values.reshape(-1, len(self.State_Parameters))
                self.Rf          = self.Data[kwargs['Risk_Free_Key']].values.reshape(-1, 1)


        # Now check if any of the state data needs overloading with the default data
        elif ('State_Parameters' in kwargs.keys()) or ('Return_Key' in kwargs.keys()) or ('Risk_Free_Key' in kwargs.keys()):

            assert not isinstance(kwargs["Return_Key"], list), 'Multiple assets may not be used with default database.'

            self.State_Parameters = kwargs['State_Parameters'] if 'State_Parameters' in kwargs.keys() else self.State_Parameters
            Return_Key            = kwargs['Return_Key']       if 'Return_Key'       in kwargs.keys() else 'Fama Mkt Excess'
            Risk_Free_Key         = kwargs['Risk_Free_Key']    if 'Risk_Free_Key'    in kwargs.keys() else '1M TBill'

            self.Data = pd.read_csv(_Default_Filename)
            self.Data = self.Data[self.State_Parameters + [Return_Key, Risk_Free_Key]]

            self.Data = self.Data.dropna()
            self.Data.reset_index(drop = True, inplace = True)

            self.Return_Data = self.Data[Return_Key].values.reshape(-1,1)
            self.State_Data  = self.Data[self.State_Parameters].values.reshape(-1, len(self.State_Parameters))
            self.Rf          = self.Data[Risk_Free_Key].values.reshape(-1,1)


        self.action_space = gym.spaces.Box(low = np.array([self.Min_Leverage] * self.Return_Data.shape[1]), high = np.array([self.Max_Leverage] * self.Return_Data.shape[1]), dtype = np.float32)
        self.observation_space = gym.spaces.Box(low = np.array([-np.inf] * (self.State_Data.shape[1] + 2)), high = np.array([np.inf] * (self.State_Data.shape[1] + 2)), dtype = np.float32)


        for key in kwargs.keys():
            if not key in ('Risk_Aversion', 'Episode_Length', 'Max_Leverage', 'Min_Leverage', 'Validation_Frac', 'Intermediate_Reward', 'DataBase', 'State_Parameters', 'Return_Key', 'Risk_Free_Key', 'Time_Step'):
                print("Keyword:", key, "not recognised.")

        if "Time_Step" in kwargs.keys() and not 'DataBase' in kwargs.keys():
            warnings.warn("Time_Step may not be changed unless the database is overridden")

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
            self.Validation_Start = np.random.randint(low = self.Episode_Length, high = (self.State_Data.shape[0] * (1 - self.Validation_Frac)) - self.Episode_Length)
            self.Validation_End   = self.Validation_Start + int(self.State_Data.shape[0] * self.Validation_Frac)
        else:
            self.Validation_Start = int(self.State_Data.shape[0] * (1 - self.Validation_Frac))
            self.Validation_End   = int(self.State_Data.shape[0])

        Mean = np.mean(np.vstack((self.Return_Data[0:self.Validation_Start], self.Return_Data[self.Validation_End:])), axis = 0)
        Vars = np.var(np.vstack((self.Return_Data[0:self.Validation_Start], self.Return_Data[self.Validation_End:])), axis = 0)

        self.Training_Var    = Vars
        self.Training_Mean   = Mean
        self.Training_Merton = self.Merton_Fraction()


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
            Possible_Indexes = np.append(np.arange(self.Validation_Start - self.Episode_Length), np.arange(self.Validation_End, self.State_Data.shape[0] - self.Episode_Length))
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

        Return = (1 + self.Rf[self.Index][0] + np.sum(self.Return_Data[self.Index] * Action))

        if self.Intermediate_Reward == True and self.Risk_Aversion == 1:
            self.Reward = self.Utility(self.Wealth * Return) - self.Utility()
        else:
            self.Reward = 0

        self.Index += 1
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

        Tau = (self.End_Index - self.Index) / (1 / self.Time_Step)
        return np.append([self.Wealth, Tau], self.State_Data[self.Index])


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
                'Rfree'  : self.Rf[self.Index]}

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

        if self.Return_Data.shape[1] == 1:
            return self.Training_Mean[0] / ((self.Training_Var[0]) * self.Risk_Aversion)

        else:
            Cov = np.cov(np.vstack((self.Return_Data[0:self.Validation_Start], self.Return_Data[self.Validation_End:])), rowvar = False)
            Data = {'Mean'  : self.Training_Mean,
                    'Var'   : self.Training_Var,
                    'Cov'   : Cov}

            cons = [{'type': 'ineq', 'fun': lambda x:  np.sum(x) - 1},
                    {'type': 'ineq', 'fun': lambda x: -np.sum(x) + 1}]

            Weights = np.array(minimize(Sharpe_Ratio, [1 / Data['Mean'].size] * Data['Mean'].size, args = (Data), constraints = cons).x).reshape(-1,1)
            Var = np.matmul(np.matmul(Weights.T, Cov), Weights)[0,0]
            Merton_Leverage = (np.sum(Weights * Data['Mean'])) / (self.Risk_Aversion * Var)

            return Weights * Merton_Leverage


    def Validate (self, N_Episodes, Agent):
        '''
        A validation function used to appraise the performance of an agent across N episodes.

        Parameters
        ----------
            N_Episodes | int
                The number of episodes to validate across.

            Agent | A compatible AC or DQN Agent.
                The Agent to validate.

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
                State, Reward, Done, Info = self.step(Action.flatten())
                Returns.append(list(Info['Mkt-Rf']))
                RF_Returns.append(Info['Rfree'][0])

                if Done:
                    Merton_Return = 1
                    RFree_Return  = 1
                    Returns = np.array(Returns)
                    for i in range(Returns.shape[0]):
                        RFree_Return  *= (1 + RF_Returns[i])
                        Merton_Return *= (1 + RF_Returns[i] + np.sum(Returns[i] * self.Training_Merton))

                    Risk_Free_Rewards.append(self.Utility(RFree_Return))
                    Merton_Rewards.append(self.Utility(Merton_Return))
                    Terminal_Rewards.append(Reward)


        self.isTraining = True
        return Terminal_Rewards, Risk_Free_Rewards, Merton_Rewards



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
