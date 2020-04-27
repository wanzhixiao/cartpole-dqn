# cartpole-dqn
this repository contains a atari game named cartpole that implement with tensorflow. more detail about cartpole can be avaliable at 
https://gym.openai.com/envs/CartPole-v0/.

a short description of deep Q-network(dqn):
this is a learning case of reinforcemnt learning algorithm deep Q-network,which mainly consits of two componet,an eval network and a 
target network. deep Q-network use replay buffer to replay exisit game playing experience obtained by a e-greedy policy.the input of
dqn is the state of environment,and it's output is the probability distribution of actions.
