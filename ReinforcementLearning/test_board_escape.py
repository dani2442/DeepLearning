from DynamicProgramming.BoardEscape.BoardEscapeModel import *

model = BoardEscapeModel(4)
model.Train()
P=model.GetArgmaxPolicy()
print(P)
