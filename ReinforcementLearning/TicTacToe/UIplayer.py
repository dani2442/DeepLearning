from Agent import Agent
class UIplayer(Agent):
	def __init__(self,player_id=None):
		self.id=player_id

	def Action(self,state,reward): 
		return int(input())
		
	def SetPlayer(self,id): self.id=id
