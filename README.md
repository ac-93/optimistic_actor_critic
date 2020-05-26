| Warning: early version, may contain bugs. |
| --- |

# optimistic-actor-critic

This repo consists of a tensorflow implementation of the Optimistic Actor Critic algorithm [1], based off the Spinningup implementation of the Soft Actor-Critic algorithm with modifications to allow for both image observations and discrete actions. Testing for the discrete case is ongoing, more detail is given below.


### Installation ###

```
# requires python>=3.6

# clone and install the repo (this may take a while)
git clone https://github.com/ac-93/optimistic-actor-critic.git
cd optimistic_actor_critic
pip install -e .

# install Spinningup from openAI
git clone https://github.com/openai/spinningup.git
cd spinningup
pip install -e .

# test the installation by running a training script
python array_observation/oac_cont/oac.py

```

### Implementation of the Optimistic Actor Critic Algorithm ###

1. Ciosek, K., Vuong, Q., Loftin, R. and Hofmann, K., 2019. Better Exploration with Optimistic Actor Critic. In Advances in Neural Information Processing Systems (pp. 1785-1796).

### Discrete Implementation ###

In soft actor critic we use pessimistic Q values (Q_LB) for choosing are actions, this is done to avoid overestimation of the value of a state action pair. However, optimistic actor critic suggests that we can find reasonable performance improvements using an optimistic Q value (Q_UB) when choosing actions.

For optimistic actor critic algorithm we would like to create a new optimistic policy that maximises the expected value  

![\mathbb{E}_{a~P_{opt}}(Q_{UB}, a)](https://render.githubusercontent.com/render/math?math=%5Cmathbb%7BE%7D_%7Ba~P_%7Bopt%7D%7D(Q_%7BUB%7D%2C%20a))

subject to the constraint

![KL(P_{opt},P_{pess}) < \delta](https://render.githubusercontent.com/render/math?math=KL(P_%7Bopt%7D%2CP_%7Bpess%7D)%20%3C%20%5Cdelta)

which gives a parameter \delta that controls how different the optimistic policy can be from the pessimistic policy.

When using continuious actions some nice maths allows for an analytic solution that gives a 1 step jump to the new optimistic value (check the paper, Appendix A1, for details). As far as I'm aware, this is not possible when using discrete actions. In place of this we can use a few optimisation steps to push the pessimistic action probabilities towards a more optimistic distribution. 

To do this we create a penalised loss function J=(expected_value)-(KL_penalty) and maximise using the ADAM optimiser for a number of steps. As this is done every environment step this causes the algorithm to slow down significantly.

In the discrete case the expected value is given by

![\mathbb{E}_{a~P_{opt}}(Q_{UB}, a) = \sum_{a}P_{opt}(a)Q_{UB}(s,a)](https://render.githubusercontent.com/render/math?math=%5Cmathbb%7BE%7D_%7Ba~P_%7Bopt%7D%7D(Q_%7BUB%7D%2C%20a)%20%3D%20%5Csum_%7Ba%7DP_%7Bopt%7D(a)Q_%7BUB%7D(s%2Ca))

and a penalty term can be defined by 

![\sum_{a}P_{opt}(a)ln(\frac{P_{opt}(a)}{P_{pess}(a)}) - \delta.](https://render.githubusercontent.com/render/math?math=%5Csum_%7Ba%7DP_%7Bopt%7D(a)ln(%5Cfrac%7BP_%7Bopt%7D(a)%7D%7BP_%7Bpess%7D(a)%7D)%20-%20%5Cdelta.)

To ensure that the penalty term and the expected value are within similar ranges for the optimisation algorthim we use a softmax on the upper bound Q values. Also, to ensure that the penalty term is only incorporated into the loss function when delta exceeded we use a relu function such that the penalty = 0 if KL(P_opt, P_pess) < delta.

An example script for running this optimisation is given in 'discrete_optimistic_policy_example.py'. 


### Main Results ### 
... to be continued
