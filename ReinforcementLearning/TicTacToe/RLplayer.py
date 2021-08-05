from Agent import Agent
class RLplayer(Agent):
	def __init__(self,player_id=None): 
		self.id=player_id
	
	def Action(self,state,reward): pass

	def SetPlayer(self,id): self.id=id