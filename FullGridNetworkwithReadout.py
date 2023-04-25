import numpy as np
import NoisyGridNetwork as gn

class FullGridNetworkwithReadout:
    def __init__(self,N,M,Rl,R,lvalues):
        self.R = R # Number of readout cells
        self.Rl = Rl # Distance
        #self.readoutPreferred = np.linspace(0, self.Rl, self.R) #Preferred Location of the Readout cell
        stepSize = self.Rl/(self.R+1)
        self.readoutPreferred = np.arange(0,Rl,stepSize)
        self.readoutPreferred = self.readoutPreferred[1:self.R+1]

        # grid cell networks init
        self.gridNets = []
        for i in range(N):
            network=gn.GridNetworkNoisy(M,lvalues[i])
            network.generateNoise(100)
            self.gridNets.append(network)

        self.M = M # Number of neurons per network
        self.N = N # Number of networks

        self.sigmah = 0.06 #Constant for G variance

        self.W = self.Weights()
    

    #Returns the value of the G function at location x with mean and variance sigma^2.
    def G(self,x):    
        return (1/(self.sigmah * np.sqrt(2 * np.pi))) * np.exp(-np.square(x)/(2*np.square(self.sigmah)))
    
    #Returns the locally peaked response of the readout cells for a location x
    def readoutTermforWeights(self,x,i):
        ri = self.G(np.abs(x-self.readoutPreferred[i]))
        return ri
    
    def Weights(self):
        W = np.zeros((self.R,self.N,self.M))
        for i in range(self.R):
            for a in range(self.N):
                for j in range(self.M):
                    for x in np.linspace(0,self.Rl,500):
                        W[i][a][j] += self.readoutTermforWeights(x,i)*self.gridNets[a].r_error_free(x,j)
        return W

    def summedInputstoReadout(self,i,x,t):
        hi=0
        temp=0
        for a in range(self.N):
            for j in range(self.M):
                hi += self.W[i][a][j]*((self.gridNets[a].r(x,j,t)+self.errorCorrectiontoGridcell(a, j, x)))
        return (hi/200)    

    def summedInputstoReadout_error_free(self,i,x):
        hi=0
        for a in range(self.N):
            for j in range(self.M):
                hi += self.W[i][a][j]*self.gridNets[a].r_error_free(x,j)
        return hi

    #Returns which Readout cell is the Winner that took all
    #h contains all the inputs to the readout cell (h=hi where i=0:R)
    def Readout(self,h):
        return np.argmax(h)

    
    def errorCorrectiontoGridcell(self,a,j,x):
        g=0
        for i in range(self.R):
            g +=self.W[i][a][j]*self.readoutTermforWeights(x,i)
        return g