# Portfolio_Gym
> A collection of OpenAI gym standard environments which may be used to facilitate training Reinforcement Learning Agents for portfolio management.  
- - - -

## Installation
The package may be installed by navigating to the folder entitled “Portfolio_Management” in terminal of cmd prompt and running the command `pip install -e .`
- - - -

## Use
Once installed the package may be imported into a Jupyter Notebook or any other python file per `import Portfolio_Gym`
The gym environment may then be instanced using gym.make with the following keywords:
1. `Simulated-v0`  - A synthetic environment which generates random returns for a single asset or a basket of correlated assets.
2. `Historical-v0` - An environment which samples historical data for episode returns. The advantage of this is that the state space of this environment may be expanded to include economic variables.
- - - -

## Parameters
`Simulated-v0`
1. **Mu** - The mean return of the asset. May be passes as either a float if you wish to run the environment for a single asset, or a numpy array of Mus to instance the environment with multiple assets.
2. **Sigma** - The standard deviation of assets returns. Must have the same type and shape as Mu.
3. **Row**    - A correlation matrix for the assets. Must be square and symmetrical. May be ignored if only one asset is used.
4. **Risk_Aversion** - The agents risk aversion, used only when calculating utility (which is the reward function).
5. **Rf** - The risk free rate.
6. **Min_Leverage** - The smallest leverage which the agent may take. Must be negative to allow the agent too short.
7. **Max_Leverage** - The highest leverage that the agent may take
8. **Time_Horizon** - The investment horizon of an episode (Years)
9. **Time_Step** - The length of a single step (Years)
10. **Model** - The model to use to generate returns. Currently the only acceptable argument is “GBM”.
11. **Intermediate_Rewards** - A flag indicating whether the environment will return intermediate rewards. Note for this parameter to have an effect Risk_Aversion must equal one, or it will default to false.
- - - -

`Historical-v0`
1. **Time_Step** - Options include "Daily" or "Monthly", refers to the database to use for training.
2. **Episode_Length** - The length of an episode, measured in Time_Step(s)
3. **Risk_Aversion** - The agents risk aversion, used only when calculating utility (which is the reward function).
4. **Min_Leverage** - The smallest leverage which the agent may take. Must be negative to allow the agent too short.
5. **Max_Leverage** - The highest leverage that the agent may take
