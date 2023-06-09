#Basic grid cell response
import numpy as np

#Noisy network of grid cell neurons
#Defined by 
#   M - Number of neurons in the network
#   l - Spatial period (lambda)
# s is the phase width (sigma) of the neural tuning curve, held constant to match paper
# s_n is standard deviation of noise
class GridNetworkNoisy:
    def __init__(self,neuronNumber,spatialPeriod):
        self.M = neuronNumber
        self.l = spatialPeriod
        self.s = .11
        self.Noise = []
        self.s_n = .04

        #self.preferredPhase = np.zeros(self.M)
        #for i in range(self.M):
        #    self.preferredPhase[i] = self.getPreferredPhase(i)
        

    def generateNoise(self,numSteps):
        self.Noise = np.zeros(numSteps)
        Inoise = 0
        for i in range(numSteps-1):
            Rnoise = np.random.normal(0,self.s_n,1)
            Inoise = Inoise + np.random.normal(0,self.s_n,1)
            self.Noise[i+1] = Rnoise+Inoise
        
    #Returns the phase response of the network for all x at time step t
    def phi(self,x,tstep):
        ideal =  x/self.l
        return np.mod(ideal+self.Noise[tstep],1)
    
    #Returns the neural tuning curve of one neuron in the network for some distance x
    def r(self,x,neuronNum,tstep):
        phiPref = self.getPreferredPhase(neuronNum)
        a = self.phi(x,tstep) - phiPref
        b = np.minimum(np.abs(a),1-np.abs(a))
        return np.exp(-np.square(b)/(2*np.square(self.s)))

    #Returns the preferred phase of neuron i 
    #Assume evenly distributed over spatial period
    def getPreferredPhase(self,neuronNum):
        stepSize = 1/(self.M+1)
        preferredPhases = np.arange(0,1,stepSize)
        preferredPhases = preferredPhases[1:self.M+1]
        return preferredPhases[neuronNum]
    
    def phi_error_free(self,x):
        return np.mod(x/self.l,1)
    
    def r_error_free(self,x,neuronNum):
        phiPref = self.getPreferredPhase(neuronNum)
        a = self.phi_error_free(x) - phiPref
        b = np.minimum(np.abs(a),1-np.abs(a))
        return np.exp(-np.square(b)/(2*np.square(self.s)))