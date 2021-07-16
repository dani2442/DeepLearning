from numpy.random import Generator, PCG64
class Initializer(object):
    def Init(): raise Exception("Function not implemented")

    @staticmethod
    def SetGenerator(n): Initializer.rng=PCG64(n)

    rng=Generator(PCG64(123)) # Generator for random numbers


from Initializers.Normal import *
from Initializers.Uniform import *
from Initializers.Xavier import *