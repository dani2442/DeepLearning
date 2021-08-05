def p(v): return ' '  if v==0 else str(int((v+1)/2))
import numpy as np
class TicTacToe(object):
	
	def __init__(self,state=None):
		if state==None:
			self.state=np.zeros((3,3),dtype=int)
		else:
			self.state=state

	def PrintBoard(self):
		s=self.state
		print(p(s[0,0]) + '|' + p(s[0,1]) + '|' + p(s[0,2]))
		print('-+-+-')
		print(p(s[1,0]) + '|' + p(s[1,1]) + '|' + p(s[1,2]))
		print('-+-+-')
		print(p(s[2,0]) + '|' + p(s[2,1]) + '|' + p(s[2,2]))

	def Start(self,Player1,Player2):
		p=[]
		r=np.random.random_integers(0,1)
		if r:
			p=[Player1,Player2]
		else:
			p=[Player2,Player1]

		p[0].SetPlayer(-1)
		p[1].SetPlayer(1)
		
		self.PrintBoard()
		print("Player {} starts".format(r))
		
		for i in range(9):
			id=2*(i%2)-1
			index=p[i%2].Action(self.state,None)
			index=self.Move(index,id)
			self.PrintBoard()
			if self.HasWon(index): 
				print("Player {} has won".format(id))
				return id
		print("Draw")
		return 0

	def Move(self,index,player):
		if type(index)==int: i=index//3;j=index%3; index=(i,j)
		if self.state[index]!=0: raise Exception("Incorrect Move")
		self.state[index]=player
		return index

	def GetMovements(self):
		pass

	def HasWon(self,index):
		i,j=index
		s=self.state
		if s[index]==s[i-1,j] and s[index]==s[i-2,j]: return True
		if s[index]==s[i,j-1] and s[index]==s[i,j-2]: return True
		if s[index]==s[i-1,j-1] and s[index]==s[i-2,j-2]: return True
		if s[index]==s[i-1,j-2] and s[index]==s[i-2,j-1]: return True
		
