#Basic grid cell response
import numpy as np

#Noiseless network of grid cell neurons
#Defined by 
#   M - Number of neurons in the network
#   l - Spatial period (lambda)
# s is the phase width (sigma) of the neural tuning curve, held constant to match paper
class GridNetworkNoiseless:
    def __init__(self,neuronNumber,spatialPeriod):
        self.M = neuronNumber
        self.l = spatialPeriod
        self.s = .11

    #Returns the ideal phase response of the network for some location x
    def phi(self,x):
        return np.mod(x/self.l,1)
    
    #Returns the neural tuning curve of one neuron in the network for some distance x
    def r(self,x,neuronNum):
        phiPref = self.getPreferredPhase(neuronNum)
        a = self.phi(x) - phiPref
        b = np.minimum(np.abs(a),1-np.abs(a))
        return np.exp(-np.square(b)/(2*np.square(self.s)))

    #Returns the preferred phase of neuron i 
    #Assume evenly distributed over spatial period
    def getPreferredPhase(self,neuronNum):
        preferredPhases = np.linspace(0,self.l,self.M)/self.l
        return preferredPhases[neuronNum]

    
