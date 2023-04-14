import numpy as np


class ReadoutCell:
    def __init__(self):

        self.R  = neuron.M*neuron.N*5 #No of ReadoutCells
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
        ri = self.G(x)*np.abs(x-self.readoutPreferred(i-1))
        return ri


    #Returns the correct activity pattern of the grid cells for a location x
    def gridcellTermforWeights(self,x,neuronNum,networkNum):
        # Use alpha term for knowing which network the neuron is in when building larger network
        if Neuron.r(x,neuronNum) == 1:
            return 1
        else:
            return 0

    # Returns the Weights for Inference
    def Weights(self,readoutNum,neuronNum,networkNum):
        for x in self.readoutPreferred:
            ReadoutcellWeights += readoutTermforWeights(x,readoutNum)*gridcellTermforWeights(x,neuronNum,networkNum)
        return ReadoutcellWeights
    

    #Returns an element of summed input to the readout cell
    def summedInputstoReadout(self,readoutNum):
        for a in self.N:
            for j in self.M:
                hi += Weights(readoutNum,j,a)*neuron.r(self.readoutPreferred(readoutNum),j)
        return hi


    #Returns which Readout cell is the Winner that took all
    #h contains all the inputs to the readout cell (h=hi where i=0:R)
    def WinnerReadout(self,h):
        return np.argmax(h)+1
    #Eventually by having the winner readout cell, the preferred location of it is our current location!