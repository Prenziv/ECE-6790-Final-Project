import numpy as np


class ReadoutCell:
    def __init__(self,N,M):

        # self.R  = neuron.M*neuron.N*5 #No of ReadoutCells
        self.N=N # network num
        self.M=M # neuron num per network
        self.R  = M*N*5 #No of ReadoutCells
        self.Rl = self.R/10  #Measure of location representation in meters
        self.readoutPreferred = np.linspace(0, self.Rl, self.R) #Preferred Location of the Readout cell

        #For finding Guassian curve to evaluate Weights
        self.mean = np.mean(self.readoutPreferred)
        self.sigmah = .06 #Not sure the value is correct 


    #Returns the value of the G function at location x with mean and variance sigma^2.
    def G(self,x):    
        return 1 / (self.sigmah * np.sqrt(2 * np.pi)) * np.exp(-(x - self.mean)**2 / (2 * self.sigmah**2))


    #Returns the locally peaked response of the readout cells for a location x
    def readoutTermforWeights(self,x,i):
        ri = self.G(x)*np.abs(x-self.readoutPreferred[i-1])
        return ri


    #Returns the correct activity pattern of the grid cells for a location x
    def gridcellTermforWeights(self,x,neuronNum,networkNum,gridNets,t):
        # Use alpha term for knowing which network the neuron is in when building larger network
        return gridNets[networkNum].r_error_free(x,neuronNum)

    # Returns the Weights for Inference
    def Weights(self,readoutNum,networkNum,neuronNum,gridNets,t=0):
        ReadoutcellWeights=0
        for x in self.readoutPreferred:
            ReadoutcellWeights += self.readoutTermforWeights(x,readoutNum)*self.gridcellTermforWeights(x,neuronNum,networkNum,gridNets,t)
        return ReadoutcellWeights
    

    #Returns an element of summed input to the readout cell
    def summedInputstoReadout(self,readoutNum,gridNets,t=0):
        hi=0
        for a in range(self.N):
            for j in range(self.M):
                hi += self.Weights(readoutNum,a,j,gridNets,t)*gridNets[a].r(self.readoutPreferred[readoutNum],j,t)
        return hi


    #Returns which Readout cell is the Winner that took all
    #h contains all the inputs to the readout cell (h=hi where i=0:R)
    def Readout(self,h):
        return np.argmax(h)
    #Eventually by having the winner readout cell, the preferred location of it is our current location!
    
    def summedInputstoReadout_error_free(self,readoutNum,gridNets):
        hi=0
        for a in range(self.N):
            for j in range(self.M):
                hi += gridNets[a].r_error_free(self.readoutPreferred[readoutNum],j)
        return hi