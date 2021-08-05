import Bandit
from Bandit.BanditAgent import BanditAgent
from Bandit.BanditEnvironment import BanditEnvironment
from Model import Model

agent = BanditAgent(5)
env = BanditEnvironment(5)

model=Model(agent,env)
model.Train(10000,verbose=40)
print(env.r)
print(agent.Q)