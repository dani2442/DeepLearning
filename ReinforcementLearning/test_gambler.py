from DynamicProgramming.Gambler.GamblerModel import GamblerModel
import matplotlib.pyplot as plt

model=GamblerModel(0.45)
model.Train(100)

#plt.plot(model.V)
plt.plot(model.P)
plt.show()