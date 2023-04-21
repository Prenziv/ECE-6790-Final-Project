import numpy as np


class ReadoutCell:
    def __init__(self,N,M, gridNets):

        # self.R  = neuron.M*neuron.N*5 #No of ReadoutCells
        self.N=N # network num
        self.M=M # neuron num per network
        self.R  = M*N*5 #No of ReadoutCells
        self.Rl = self.R/10  #Measure of location representation in meters
        self.readoutPreferred = np.linspace(0, self.Rl, self.R) #Preferred Location of the Readout cell

        #For finding Guassian curve to evaluate Weights
        self.mean = np.mean(self.readoutPreferred)
        self.sigmah = .6 #Not sure the value is correct 
        self.gridNets=gridNets
        self.Weights_array= self.Weights()
        


    #Returns the value of the G function at location x with mean and variance sigma^2.
    def G(self,x,readoutNum):    
        return 1 / (self.sigmah * np.sqrt(2 * np.pi)) * np.exp(-(x - (readoutNum/10))**2 / (2 * self.sigmah**2))


    #Returns the locally peaked response of the readout cells for a location x
    def readoutTermforWeights(self,x,i):
        #ri = self.G(x,i)*np.abs(x-self.readoutPreferred[i-1])
        ri = self.G(np.abs(x-self.readoutPreferred[i-1]),i)
        print(ri)
        return ri


    #Returns the correct activity pattern of the grid cells for a location x
    def gridcellTermforWeights(self,x,neuronNum,networkNum,gridNets):
        # Use alpha term for knowing which network the neuron is in when building larger network
        if gridNets[networkNum].r_error_free(x,neuronNum)==1:
            return 1
        else:
            return 0

    # Returns the Weights for Inference
    def Weights(self):
        ReadoutcellWeights=np.zeros((self.R,self.N,self.M))
        for i in range(self.R):
            for j in range(self.N):
                for p in range(self.M):
                    for x in self.readoutPreferred:
                        ReadoutcellWeights[i][j][p] += self.readoutTermforWeights(x,i)*self.gridcellTermforWeights(x,p,j,self.gridNets)
        #print (ReadoutcellWeights)
        return ReadoutcellWeights
    

    #Returns an element of summed input to the readout cell
    def summedInputstoReadout(self,readoutNum,gridNets,X,t=0):
        hi=0
        for j in range(self.M):
            for a in range(self.N):
                hi += self.Weights_array[readoutNum][a][j]*gridNets[a].r(X,j,t)
                #print(gridNets[a].r(X,j,t))
                # print(self.Weights(readoutNum,a,j,gridNets))
        return hi


    #Returns which Readout cell is the Winner that took all
    #h contains all the inputs to the readout cell (h=hi where i=0:R)
    def Readout(self,h):
        return np.argmax(h)
    #Eventually by having the winner readout cell, the preferred location of it is our current location!
    
    def summedInputstoReadout_error_free(self,readoutNum,gridNets,X):
        hi=0
        for j in range(self.M):
            for a in range(self.N):
                hi += self.Weights_array[readoutNum][a][j]*gridNets[a].r_error_free(X,j)
        return hi