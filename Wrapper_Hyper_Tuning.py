import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp
import itertools
import random
import tensorflow as tf

from copy import deepcopy
from functools import partial
from time import sleep

def train_and_collect_test(specs, agent, N_eps):
    '''
        Parameters
        ----------
            Agent           | class
                Unconstructed actor critic worker
            N_eps           | int
                Number of episodes to train the agent before making OOS test
        Returns
        -------
            Result          | dict
                A dictionary containing performance different between agent and merton model, percentage difference and sharpe ratio of agent
            
        Notes
        -----
        Defining the result collection function at the top level to make it pickable.
        This is the function to be paralleled; pickable is the requirement of multiprocessing package 
        
    '''
    Agent = agent(**specs)
    Agent.Train(N_Episodes = N_eps)
    Terminal_Rewards, Risk_Free_Rewards, Merton_Rewards = Agent.Environment.Validate(100, Agent)
    Mean_Terminal   = sum(Terminal_Rewards)  /len(Terminal_Rewards)
    Mean_Merton     = sum(Merton_Rewards)    /len(Merton_Rewards)
    Mean_RF         = sum(Risk_Free_Rewards) /len(Risk_Free_Rewards)
    mean_Delta      = Mean_Terminal - Mean_Merton
    pct_Delta       = Mean_Terminal/Mean_Merton - 1
    sharpe          = (Mean_Terminal - Mean_RF)/np.std(Terminal_Rewards)
    return {'Average Improve': mean_Delta, 'Percentage Improve': pct_Delta, 'Sharpe Ratio' : sharpe}


class Wrapper_WIP:
    
    def __init__(self, Environment, Agent):
        '''
        Parameters
        ----------
            Agent           | class
                Unconstructed object of an Actor-Critic agent
                
            Environment     | class
                Constructed object of environment
        '''

        self.N_core = mp.cpu_count()
        self.Environment = Environment 
        self._Agent      = Agent
        
    
    def Set_Params(self, Target, N_Episodes, Agent_Specs, **kwargs):
        '''
        Setting target for this parameter tuning then create scenarios according to different target. 
        Parameters
        ----------
        
            Target      | str 
                One of following four options:           
                - Repeat_Env: Train multiple instances on the same environment to confirm it can cosistantly learn irregard of different initial network weights.
                - Identical_Hyper: Train multiple instances of agents with same hyper parameter on different instances of environments, to assess of impact of environment on agents' performance
                - Para_Grid_Search: Tune hyper parameters using a grid search over all combinations of discretised hyper parameters
                - Genetic Algorithm: Tune hyper parameter using a survive-mutate process to find the optimal parameter over N-generations
                When use genetic algorithm, we adopt a simple stopping point of N-generations, future versions may use parameter-delta or performance-delta between generations

            N_Episodes  | int
                Number of Episodes used to train each agent before put to test in hold-out period
            
            Agent_Specs | dict
                Default settings of agent parameters
            
            N_cores     | int
                (optional) how many cores of the machine to use
            
            Para_Range  | dict
                (Para_Grid_Search) when use grid search, user should supply dictionary of hyper parameters' spaces in a dictionary
            
            Population  | list
                (Genetic_Algorithm) when use Genetic Algorithm, user should supply a list of dictionaries defining specs of each individual agents in the population
            
            Generations | int
                (optional, Genetic_Algorithm) generations to evolve when use genetic algorithm
                
        '''
        self._Target        = Target
        self._Agent_Specs   = Agent_Specs
        self._N_Eps         = N_Episodes
 
        if 'N_cores' in kwargs:
            assert kwargs['N_cores'] <= self.N_core, 'Machine has insufficient many cores to support {} parallels'.format(kwargs['N_cores'])
            self.N_core = kwargs['N_cores']
        
        if Target  ==  'Genetic_Algorithm': 
            self._N_generations = kwargs['Generations'] if 'Generations' in kwargs else 10
        self._Create_Scenario(**kwargs)
        
        
    def _Create_Scenario(self, **kwargs):
        '''
        Create the Agents/Environments according to the discretion specified in Set_Param
        Parameters
        ----------
            Not a user function to call, same parameter as Set_Param function
            
        Returns
        -------
            None
        
        Notes
        -----
            create a list of agents according to Target:
                - Repeat_Env: 20 agents with same hyper parameter, different weight initiation, working on identical environment
                - Identical_Hyper: 20 agents with same hyper parameter, differnet weight initiation, working on different environments with same factor_setting
                - Para_Grid_Search: N agents (one for each different hyper parameter setting), working on different environments
                - Genetic_Algorithm: N agent according to the population's parameter setting working on different environments
        
        '''
        self.Spec_list = []
        
        if self._Target == 'Repeat_Env':    
            for i in range(20):
                specs = deepcopy(self._Agent_Specs)
                specs.update({'Environment': deepcopy(self.Environment),
                              'thread' : i})
                self.Spec_list.append(specs)
            self._isTrained = False
        
        
        if self._Target == 'Identical_Hyper':    
            for i in range(20):
                specs = deepcopy(self._Agent_Specs)
                Env = deepcopy(self.Environment)
                Env.Genrate_Returns()
                specs.update({'Environment':Env, 'thread': i})
                self.Spec_list.append(specs)
            self._isTrained = False
        
        
        if self._Target == 'Para_Grid_Search':
            assert 'Para_Range' in kwargs, 'Which Parameter do you want to tune?'
            para_dict = kwargs['Para_Range']
            Key, value = zip(*para_dict.items())
            _comb_values = [i for i in itertools.product(*value)]
            All_Combinations = [{Key[i] : item[i] for i in range(len(item))} for item in _comb_values]
            for i, para_pair in enumerate(All_Combinations):
                Env = deepcopy(self.Environment)
                Env.Genrate_Returns()
                specs = deepcopy(self._Agent_Specs)
                specs.update(para_pair.update({'Environment': Env, 'thread': i}))
                self.Spec_list.append(specs)
            self._isTrained = False
            
            
        if self._Target == 'Genetic_Algorithm':
            assert 'Population' in kwargs, 'Please supply surviver cases'
            para_list = kwargs['Population']
            for i, New_specs in enumerate(para_list):
                Env = deepcopy(self.Environment)
                Env.Genrate_Returns()
                specs = deepcopy(self._Agent_Specs)
                specs.update(New_specs.update({'thread':i}))
                self.Spec_list.append(specs)
            self._isTrained = False
        

    def Run(self):
        '''
        Parameters
        ----------
            None
            
        Returns
        -------
            Result          | dict
            returns a structured output of parameter tuning result
            Chart           | display
            displays the charts of parameter tuning results
        
        Notes
        -----
            Train the agents in the agents list, one or multiple times, and collect their out of sample results.
            Then generate plots and result summarises based on the collected results
        '''
        
        if self._Target == 'Genetic_Algorithm':
            while self._N_Intrapolates >=0:
                Result, Specs = self._Parallel_Process()
                Surviver = self._Selection(Result, Specs)
                if self._N_Intrapolates!= 0: self._Create_Scenario(Population = Surviver)
        else:
            Result, _ = self._Parallel_Process()
        self.isTrained = True
        output = self.Plotting_Function(Result)
        
        return output
    
    
    def _Selection(self, Result_In, Specs_In, by = 'Average Improve'):
        
        Criteria        = [Items[by] for Items in Result_In]
        surviver_idx    = np.argsort(Criteria)[-int(len(Criteria)*0.2):]       
        survived_para   = [Specs_In[i] for i in surviver_idx]
        
        New_Population = []
        # Randomly select from the survived population to produce offspring with some chance of mutation until population size
        for i in range(len(Specs_In)):
            Father, Mother  = random.sample(survived_para, 2)
            Off_spring      = {item : (Father[item] + Mother[item])/2 for item in Father}
            if np.random.uniform(0,1)<=0.1:
                Off_spring  = {item : Off_spring[item] * 2 for item in Off_spring}
            New_Population.append(Off_spring)
            
        self._N_generations -= 1
        return New_Population
    
    
    def _Parallel_Process(self):
        '''
        Parameters
        ----------
            Not a user function to call, No parameters
            
        Returns
        -------
            Results of different parameters and parameters themselves
            
        Notes
        -----
            This function runs in parallel the agents stored in agents' list. Then collect OOS result summaries, and return results with parameters
            Result either goes into 
        '''
        pool            = mp.Pool(self.N_core)
        worker_list     = [pool.apply_async(train_and_collect_test, (item, self._Agent, self._N_Eps)) for item in self.Spec_list]
        All_result      = [worker.get() for worker in worker_list]
        pool.join()
        return All_result, self.Spec_list
    
    
    def Plotting_Function (self, Result):
        # some beautiful charts and ordered illustrative results
        return Result



