import numpy             as np
import matplotlib.pyplot as plt

def Empty (*args):
    ''' An empty function which accepts any number of arguments '''
    pass


class Wrapper:

    '''
    This class is to be used to train RL Agents with either the simulated or historical portfolio gyms
    and to evalaute the ability of the agents to outperform the merton portfolio using ecconomic factors
    '''

    def __init__ (self, Agent):
        '''
        Parameters
        ----------
            Agent | A compatible DQN or AC agent
                The Agent to train.
        '''

        self.Agent = Agent


    def Train (self, N_Episodes, Plot = [], Diagnostics = []):
        '''
        Train the agent inside the wrapper so plots may be generated.

        Parameters
        ----------
            N_Episodes | int
                The number of episodes to train the agent for.

            Plots | list
                A list containing keywords for the plots to generate. Acceptable inputs include:
                    1. 'Mu' : This will return a plot of the true (pre sigma/epsilon) predictions made by the agent. It will have lenght equivalent to the total number of steps in the training episodes, and will have the same dimensions as the actionspace.

                    2. Merton_Benchmark : Plot the average utility of both the Merton Portfolio and the DQN across 100 episodes (from the training dataset) acting greedily

                    3. Percent_Merton_Action : Returns the fraction of actions recomended by the actor which are within 10% of the merton portfolio action.

        '''


        self.Plot = Plot
        self.Diagnostics = Diagnostics

        self.Plot_Data = {}
        for key in self.Plot:
            self.Plot_Data[key] = []

        for i in range(len(self.Diagnostics)):
            self.Plot_Data['Diag' + str(i)] = []

        Arg_A = self.Plotting_Function if len(self.Plot) > 0 else Empty
        Arg_B = self.Run_Diagnostics   if len(self.Diagnostics) > 0 else Empty

        self.Agent.Train(N_Episodes, Arg_A, Arg_B)

        if hasattr(self.Agent.Environment, 'Validate'):
            self.Plot_Data['Validate'] = self.Agent.Environment.Validate(100, self.Agent)

        self.Display()


    def Plotting_Function (self, Episode_Exps):

        if 'Merton_Benchmark' in self.Plot_Data.keys():
            results = {'Agent'  : [],
                       'Merton' : []}

            for episode in Episode_Exps:
                Merton_Wealth = 1
                Agent_Wealth  = 1
                for step_exp in episode:
                    Merton_Wealth *= (1 + step_exp['i']['Rfree'] + np.sum(step_exp['i']['Mkt-Rf'] * self.Agent.Environment.Training_Merton))
                    Agent_Wealth  *= (1 + step_exp['i']['Rfree'] + np.sum(step_exp['i']['Mkt-Rf'] * step_exp['Mu']))

                results['Agent'].append(self.Agent.Environment.Utility(Agent_Wealth))
                results['Merton'].append(self.Agent.Environment.Utility(Merton_Wealth))

            self.Plot_Data['Merton_Benchmark'].append([np.mean(results['Agent']), np.mean(results['Merton'])])


        if 'Percent_Merton_Action' in self.Plot_Data.keys():
            Count = 0
            Lenght = 0
            for Exp in Episode_Exps:
                Lenght += len(Exp)
                for exp in Exp:
                    if np.all(exp['Mu'] > self.Agent.Environment.Training_Merton * 0.9) and np.all(exp['Mu'] < self.Agent.Environment.Training_Merton * 1.1):
                        Count += 1

            self.Plot_Data['Percent_Merton_Action'].append((Count / Lenght) * 100)


        if 'Mu' in self.Plot_Data.keys():
            for Exp in Episode_Exps:
                for exp in Exp:
                    self.Plot_Data['Mu'].append(exp['Mu'])


    def Display (self):
        ''' Display the data '''

        if len(self.Plot_Data.keys()) == 0 : return None
        f, ax = plt.subplots(1 , len(self.Plot_Data.keys()), figsize = (6 * len(self.Plot_Data.keys()), 6))
        if len(self.Plot_Data.keys()) == 1 : ax = [ax]
        i = 0

        for j in range(len(self.Diagnostics)):
            for Data in self.Plot_Data['Diag' + str(j)]:
                ax[i].plot(Data[0], Data[1])
                ax[i].set_title('Diag: ' + self.Diagnostics[j]['Module'] + ', Factor: ' + str(self.Diagnostics[j]['Factor']))
            i += 1

        if 'Mu' in self.Plot_Data.keys():
            ax[i].plot(np.array(self.Plot_Data['Mu']), label = 'Actions')
            ax[i].set_xlabel('Step')
            ax[i].set_ylabel('Leverage')
            if isinstance(self.Agent.Environment.Training_Merton, type(np.array([0]))):
                for j in range(self.Agent.Environment.Training_Merton.size):
                    ax[i].axhline(self.Agent.Environment.Training_Merton[j], color = 'k')
            else:
                ax[i].axhline(self.Agent.Environment.Training_Merton, color = 'k')
            i += 1


        if 'Merton_Benchmark' in self.Plot_Data.keys():
            self.Plot_Data['Merton_Benchmark'] = np.array(self.Plot_Data['Merton_Benchmark'])
            ax[i].scatter(np.arange(self.Plot_Data['Merton_Benchmark'].shape[0]), self.Plot_Data['Merton_Benchmark'][:,0], label = 'Agent',  color = 'lightskyblue')
            ax[i].scatter(np.arange(self.Plot_Data['Merton_Benchmark'].shape[0]), self.Plot_Data['Merton_Benchmark'][:,1], label = 'Merton', color = 'mediumvioletred')
            ax[i].set_xlabel('Model')
            ax[i].set_ylabel('Average Terminal Utility')
            ax[i].legend()
            i += 1


        if 'Percent_Merton_Action' in self.Plot_Data.keys():
            ax[i].scatter(np.arange(len(self.Plot_Data['Percent_Merton_Action'])), self.Plot_Data['Percent_Merton_Action'], color = 'mediumvioletred')
            ax[i].set_xlabel('Model')
            ax[i].set_ylabel('Percentage')
            i += 1


        if 'R_Squared' in self.Plot_Data.keys():
            ax[i].bar(np.arange(len(self.Plot_Data['R_Squared'])), self.Plot_Data['R_Squared'], color = 'darkblue')
            ax[i].set_xlabel('Model')
            ax[i].set_ylabel('R Squared')
            i += 1


        if 'Validate' in self.Plot_Data.keys():
            ax[i].scatter(np.arange(len(self.Plot_Data['Validate'][0])), self.Plot_Data['Validate'][0], label = 'Agent',  color = 'lightskyblue')
            ax[i].scatter(np.arange(len(self.Plot_Data['Validate'][1])), self.Plot_Data['Validate'][1], label = 'RFree',  color = 'mediumvioletred')
            ax[i].scatter(np.arange(len(self.Plot_Data['Validate'][2])), self.Plot_Data['Validate'][2], label = 'Merton', color = 'darkblue')
            ax[i].set_xlabel('Validation Episode')
            ax[i].set_ylabel('Utility')
            ax[i].legend()


        plt.show()


    def Run_Diagnostics (self):

        for i, Diag in enumerate(self.Diagnostics):
            State = np.zeros((25, self.Agent.State_Dim))
            State[:,0] = 1
            State[:,1] = 0.5

            if Diag['Factor'] == 0:
                X = np.linspace(0, 2, 25)
            elif Diag['Factor'] == 1:
                X = np.linspace(0, 1, 25)
            State[:, Diag['Factor']] = X

            if Diag['Module'] == 'Actor':
                Y = self.Agent.Predict_Action(State).flatten()
            elif Diag['Module'] == 'Critic':
                Y = self.Agent.Predict_Value(State).flatten()
            else:
                warnings.warn('Diagnostics Module not recognised')
                return None

            self.Plot_Data['Diag' + str(i)].append([X, Y])

        return X, Y
