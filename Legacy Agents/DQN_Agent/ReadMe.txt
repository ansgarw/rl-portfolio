There are three versions of DQN included:

    Version IV        - The original DQN agent from which the other versions have splintered.
    Version V         - As Version IV, only with the inclusion of Asyncronous experiance gathering for mild
                        performance increase. However it needs to instance the environment itself inside the class
                        and as such does not yet support environments with many setup parameters (such as Merton)
    MultiHead version - As version IV but with discrete network output for each action.

There are also two Jupyter Notebooks which may be used to train the models to either the MountainCar or Merton environments.



ToDo:

    1.  Version V needs imporvement to allow it to work with Merton environment, however since there is no need to
        over sample positive experiance in the merton environment the performance improvements of V are minimal, and
        IV is easily sufficient.

Multi-Headed Version:

    2.  As far as I am aware it is impossible to train a Sklearn NN with multiple outputs with a dataset which has
        only one output, (if the model is multiheaded for 3 actions for example then to train the model the observed
        quality of all three actions from any state are required, however we only observe the quality of the one action
        that was taken).  Hence the mutliheaded agent currently creates independent NNs to approximate each action. This is
        not ideal as it increases the ammount of data required to fully fit the model, as the total number of parameters
        now increases linearly with the action dimensions.
    3.  This version also requires small polishings, such as additional setup arguments etc.
