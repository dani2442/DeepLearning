from Player import Player
class UIplayer(Player):
	def __init__(self,player_id=None): super().__init__(player_id)

	def Move(self,state): 
		return int(input())
		
