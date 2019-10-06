from gym.envs.registration import register

register(id = 'Simulated-v0', entry_point = 'Portfolio_Gym.envs:SimulatedEnv', kwargs = {'Mu'                  : 0.05,
                                                                                         'Sigma'               : 0.15,
                                                                                         'Row'                 : None,
                                                                                         'Risk_Aversion'       : 1,
                                                                                         'Rf'                  : 0.02,
                                                                                         'Max_Leverage'        : 5,
                                                                                         'Min_Leverage'        : 5,
                                                                                         'Time_Horizon'        : 1,
                                                                                         'Time_Step'           : 1/12,
                                                                                         'Model'               : "GBM",
                                                                                         'Intermediate_Reward' : False,
                                                                                         'State_Corrolations'  : []})


register(id = 'Historical_Daily-v0', entry_point = 'Portfolio_Gym.envs:HistoricalEnv', kwargs = {'Risk_Aversion'  : 1,
                                                                                           'Time_Step'            : "Daily",
                                                                                           'Episode_Length'       : 252,
                                                                                           'Max_Leverage'         : 10,
                                                                                           'Min_Leverage'         : 10,
                                                                                           'Validation_Frac'      : 0.3,
                                                                                           'Fama_Returns'         : True,
                                                                                           'Technical_Data'       : False,
                                                                                           'Intermediate_Reward'  : False},)

register(id = 'Historical_Monthly-v0', entry_point = 'Portfolio_Gym.envs:HistoricalEnv', kwargs = {'Risk_Aversion'       : 1,
                                                                                                   'Time_Step'           : "Monthly",
                                                                                                   'Episode_Length'      : 12,
                                                                                                   'Max_Leverage'        : 10,
                                                                                                   'Min_Leverage'        : 10,
                                                                                                   'Validation_Frac'     : 0.3,
                                                                                                   'Fama_Returns'        : True,
                                                                                                   'Technical_Data'      : False,
                                                                                                   'Intermediate_Reward' : False},)

# Depricated
register(id = 'Historical_Monthly-v1', entry_point = 'Portfolio_Gym.envs:HistoricalEnv', kwargs = {'Risk_Aversion'       : 1,
                                                                                                   'Time_Step'           : "Monthly",
                                                                                                   'Episode_Length'      : 12,
                                                                                                   'Max_Leverage'        : 10,
                                                                                                   'Min_Leverage'        : 10,
                                                                                                   'Validation_Frac'     : 0.3,
                                                                                                   'Fama_Returns'        : True,
                                                                                                   'Technical_Data'      : False,
                                                                                                   'Intermediate_Reward' : False},)
