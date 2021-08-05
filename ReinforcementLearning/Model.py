from Agent import Agent
from Environment import Environment
class Model:
    def __init__(self,agent :Agent,environment :Environment):
        self.agent=agent
        self.env=environment

        self.acum_reward=0

    def Train(self,iter=100,verbose=10):
        state,reward=self.env,None
        for i in range(iter):
            action=self.agent.Action(state,reward)
            state,reward=self.env.NextTimeStep(action)
            
            self.acum_reward+=reward
            if i%verbose==0: print(self.acum_reward/(i+1))
            