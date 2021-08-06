import numpy as np
from Bandit.BanditAgent import BanditAgent
from Bandit.BanditAgentUCB import BanditAgentUCB
from Bandit.BanditAgentGradient import BanditAgentGradient
from Bandit.BanditEnvironment import BanditEnvironment
from Model import Model

np.random.seed(12)
env = BanditEnvironment(5)

agent = BanditAgent(5,eps=0.1,l_rate=0.01)
agent2=BanditAgentUCB(5,0.2)
agent3=BanditAgentGradient(5,0.1)

model=Model(agent3,env)
model.Train(10000,verbose=40)
print(env.r)
print(agent.Q)