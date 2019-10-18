from gym.envs.registration import register
import numpy

register(id = 'Simulated-v0', entry_point = 'Portfolio_Gym.envs:SimulatedEnv', kwargs = {'Mu'                  : 0.05,
                                                                                         'Sigma'               : 0.15,
                                                                                         'Row'                 : None,
                                                                                         'Risk_Aversion'       : 1,
                                                                                         'Rf'                  : 0.02,
                                                                                         'Max_Leverage'        : 5,
                                                                                         'Min_Leverage'        : 5,
                                                                                         'Time_Horizon'        : 1,
                                                                                         'Time_Step'           : 1/12,
                                                                                         'Intermediate_Reward' : False})


register(id = 'Simulated-v1', entry_point = 'Portfolio_Gym.envs:Simulated_VAR_Env', kwargs = {'Factor_Beta'         : numpy.array([-0.1694, 0.9514]).reshape(-1,1),
                                                                                              'Asset_Beta'          : numpy.array([0.0249, 0.0568]).reshape(-1,1),
                                                                                              'Cov'                 : numpy.array([[6.225, -6.044], [-6.044, 6.316]]),
                                                                                              'Risk_Aversion'       : 1,
                                                                                              'Rf'                  : 0.02,
                                                                                              'Max_Leverage'        : 5,
                                                                                              'Min_Leverage'        : 5,
                                                                                              'Time_Horizon'        : 1,
                                                                                              'Time_Step'           : 1/12,
                                                                                              'Intermediate_Reward' : False,
                                                                                              'Factor_State_Len'    : 1})


register(id = 'Historical-v0', entry_point = 'Portfolio_Gym.envs:HistoricalEnv', kwargs = {'Risk_Aversion'        : 1,
                                                                                           'Time_Step'            : 1/12,
                                                                                           'Episode_Length'       : 12,
                                                                                           'Max_Leverage'         : 10,
                                                                                           'Min_Leverage'         : 10,
                                                                                           'Validation_Frac'      : 0.3,
                                                                                           'Intermediate_Reward'  : False},)
