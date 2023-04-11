#Basic grid cell response
import numpy as np

#Noisy network of grid cell neurons
#Defined by 
#   M - Number of neurons in the network
#   l - Spatial period (lambda)
# s is the phase width (sigma) of the neural tuning curve, held constant to match paper
class GridNetworkNoisy:
    def __init__(self,neuronNumber,spatialPeriod):
        self.M = neuronNumber
        self.l = spatialPeriod
        self.s = .11

    #Returns the phase response of the network for some location x
    #over some time t
    #Make sure length(t) == length(x) 
    def phi(self,x,t,deltat):
        ideal =  x/self.l
        Rnoise = np.random.normal(0,.04,1)
        Inoise = 0
        for i in range(int(t/deltat)):
            Inoise = Inoise + np.random.normal(0,.04,1)
        
        return np.mod(ideal+Rnoise+Inoise,1)
    
    #Returns the neural tuning curve of one neuron in the network for some distance x
    def r(self,x,neuronNum,t,deltat):
        phiPref = self.getPreferredPhase(neuronNum)
        a = self.phi(x,t,deltat) - phiPref
        b = np.minimum(np.abs(a),1-np.abs(a))
        return np.exp(-np.square(b)/(2*np.square(self.s)))

    #Returns the preferred phase of neuron i 
    #Assume evenly distributed over spatial period
    def getPreferredPhase(self,neuronNum):
        preferredPhases = np.linspace(0,self.l,self.M)/self.l
        return preferredPhases[neuronNum]