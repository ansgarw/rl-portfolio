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


    def Train (self, N_Episodes, Plot = [], Diagnostics = [], Validate = False):
        '''
        Train the agent inside the wrapper so plots may be generated. This is the only function the user needs to call.

        Parameters
        ----------
            N_Episodes | int
                The number of episodes to train the agent for.

            Plots | list
                A list containing keywords for the plots to generate. Acceptable inputs include:
                    1. 'Mu' : This will return a plot of the true (pre sigma/epsilon) predictions made by the agent.

                    2. Merton_Benchmark : Plot the average utility of both the Merton Portfolio and the Agent across training, acting greedily

                    3. Percent_Merton_Action : Returns the fraction of actions recomended by the actor which are within 10% of the merton portfolio action.

                    4. VAR_Benchmark : Plot the average utility of both the VAR Optimal and the Agent across training, acting greedily

            Diagnostics | List (of dict)
                A list contianing dictionaries identifying the required disgnostic plots. Currently only 2D plots of the sensativity of either the Actor or Critic vs one state parameter may be generated. The dictionaries must include two keys:
                    1. 'Module' | string
                        Either 'Actor' or 'Critic'
                    2. 'Factor' | int
                        The index of the state parameter whose sensitivity is desired.

                Currently no diagnostics for the DQN are implemented.

        '''

        self.N_Episodes = N_Episodes

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

        if Validate == True:
            self.Plot_Data['Validate'] = self.Agent.Environment.Validate(100, self.Agent)

        self.Display()


    def Plotting_Function (self, Episode_Exps):

        '''
        The plotting function which is passed to an Agents train function to record its performance whilst training. Not user facing.

        Parameters
        ----------
            Episode_Exps | list
                A list of lists of experiance dictionaries. Each list of expeiance dictionaries should include all of the experiance from a single episode, in the correct order.
        '''

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

        if 'VAR_Benchmark' in self.Plot_Data.keys():
            results = {'Agent'  : [],
                       'Optimal' : []}

            for episode in Episode_Exps:
                Optimal_Wealth = 1
                Agent_Wealth  = 1
                for step_exp in episode:
                    Optimal_Wealth *= (1 + step_exp['i']['Rfree'] + np.sum(step_exp['i']['Mkt-Rf'] * self.Agent.Environment.VAR_Benchmark(step_exp['s0'][2])))
                    Agent_Wealth   *= (1 + step_exp['i']['Rfree'] + np.sum(step_exp['i']['Mkt-Rf'] * step_exp['Mu']))

                results['Agent'].append(self.Agent.Environment.Utility(Agent_Wealth))
                results['Optimal'].append(self.Agent.Environment.Utility(Optimal_Wealth))

            self.Plot_Data['VAR_Benchmark'].append([np.mean(results['Agent']), np.mean(results['Optimal'])])


    def Display (self):
        '''
        Construct and display the requested plots. Not user facing.
        '''

        if len(self.Plot_Data.keys()) == 0 : return None
        f, ax = plt.subplots(1 , len(self.Plot_Data.keys()), figsize = (6 * len(self.Plot_Data.keys()), 6))
        if len(self.Plot_Data.keys()) == 1 : ax = [ax]
        i = 0

        for j in range(len(self.Diagnostics)):
            for k, Data in enumerate(self.Plot_Data['Diag' + str(j)]):
                ax[i].plot(Data[0], Data[1], label = (k / 5) * self.N_Episodes)
                ax[i].set_title('Diag: ' + self.Diagnostics[j]['Module'] + ', Factor: ' + str(self.Diagnostics[j]['Factor']))

            if self.Diagnostics[j]['Factor'] == 0 and self.Diagnostics[j]['Module'] == 'Critic':
                # Plot the true critic value in this case:
                ax[i].plot([None] + list(Data[0][1::]), [None] + list(self.Agent.Environment.Utility(Data[0][1::])), color = 'k')
            ax[i].legend()
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
            ax[i].set_xlabel('Steps (x10000)')
            ax[i].set_ylabel('Average Terminal Utility')
            ax[i].legend()
            i += 1


        if 'VAR_Benchmark' in self.Plot_Data.keys():
            self.Plot_Data['VAR_Benchmark'] = np.array(self.Plot_Data['VAR_Benchmark'])
            ax[i].scatter(np.arange(self.Plot_Data['VAR_Benchmark'].shape[0]), self.Plot_Data['VAR_Benchmark'][:,0], label = 'Agent',  color = 'lightskyblue')
            ax[i].scatter(np.arange(self.Plot_Data['VAR_Benchmark'].shape[0]), self.Plot_Data['VAR_Benchmark'][:,1], label = 'Optimal', color = 'mediumvioletred')
            ax[i].set_xlabel('Steps (x10000)')
            ax[i].set_ylabel('Average Terminal Utility')
            ax[i].legend()
            i += 1


        if 'Percent_Merton_Action' in self.Plot_Data.keys():
            ax[i].scatter(np.arange(len(self.Plot_Data['Percent_Merton_Action'])), self.Plot_Data['Percent_Merton_Action'], color = 'mediumvioletred')
            ax[i].set_xlabel('Steps (x10000)')
            ax[i].set_ylabel('Percentage')
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

        '''
        The plotting function which is passed to an agents train function to facilitate the construction of diagnostics plots. The performance evalaution plots are called much more frequently, hence these plots needed to be segregated.

        Currently supports the construction of 2D plots of the sensitivity of the Actor or Critic to one of the state parameters, holding all others constant at one.

        Notes
        -----
            It does not make sense to use this with environments with many factors, as each factor will have its own range which might not include one, hence the plots will no longer be a true depiction of the sensitivity of the networks. Further work into depicting network sensitivities required.
            That said these functions were instrumental to understanding the AC performance in the simple simulation.
        '''

        for i, Diag in enumerate(self.Diagnostics):
            State = np.zeros((25, self.Agent.State_Dim))
            State[:,0] = 1
            State[:,1] = 0.5
            State[:,2] = -3.5

            if Diag['Factor'] == 0:
                X = np.linspace(0, 2, 25)
            elif Diag['Factor'] == 1:
                X = np.linspace(0, 1, 25)
            elif Diag['Factor'] == 2:
                X = np.linspace(-4,-3,25)
            else:
                X = np.linspace(0,1,25)

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
