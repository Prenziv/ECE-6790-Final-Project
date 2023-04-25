#Testbench
import NoisyGridNetwork as gn
import numpy as np
import matplotlib.pyplot as plt
import FullGridNetworkwithReadout as cn

#Removes discontinuities when plotting to avoid sharp lines
def removeDiscontinuity(x,y):
    for i in range(len(y)):
        if np.abs(y[i] - y[i-1]) > .5:
            y = np.insert(y,i,np.nan)
            x = np.insert(x,i,np.nan)
    
    return x,y

#Finds the nearest value in the array to the given value
def find_nearest(array, values):
    indices = np.abs(np.subtract.outer(array, values)).argmin(0)
    return indices

#Plots 
# - Phase of network over some spatial periods and 3 time steps
# - Neural tuning curve of one neuron over 3 spatial periods and 3 time steps
# Uses noisy neuron case
def gridCellPlot(network,numPeriods):
    #Generate noise for 10 time steps
    network.generateNoise(10)

    numPoints = 250*numPeriods

    x = np.linspace(0,numPeriods*network.l,numPoints)

    plt.figure()
    plt.subplot(211)
    phaseResponse = network.phi_error_free(x)
    [xnew,ynew] = removeDiscontinuity(x,phaseResponse)
    plt.plot(xnew,ynew,linewidth=2,color='black')

    phaseResponse = network.phi(x,3)
    [xnew,ynew] = removeDiscontinuity(x,phaseResponse)
    plt.plot(xnew,ynew,linewidth=2,color='tab:blue')

    phaseResponse = network.phi(x,9)
    [xnew,ynew] = removeDiscontinuity(x,phaseResponse)
    plt.plot(xnew,ynew,linewidth=2,color='tab:red')

    plt.yticks([0,1])
    plt.ylim([0,1])
    plt.xlim([0,numPeriods*l])
    plt.tick_params(bottom = False,labelbottom=False)

    plt.subplot(212)
    r = network.r_error_free(x,int(M/2))
    plt.plot(x,r,linewidth=2,color='black')

    r = network.r(x,int(M/2),3)
    plt.plot(x,r,linewidth=2,color='tab:blue')

    r = network.r(x,int(M/2),9)
    plt.plot(x,r,linewidth=2,color='tab:red')

    plt.yticks([0,1])
    plt.ylim([0,1])
    plt.xlim([0,numPeriods*l])
    plt.tick_params(bottom = False,labelbottom=False)
    plt.show()

#Plots neural tuning curve response of all neurons over two spatial periods
#for two full grid cell networks of the same size with different spatial periods
#in their individual networks
def plotR(N,M,lvalues1,lvalues2):    
    numPoints = 1000

    maxPoint = max([lvalues1+lvalues2])
    maxPoint = maxPoint[0]

    #Network 1
    plt.subplot(211)
    lvalues = lvalues1
    gridNets = []
    for i in range(N):
        network=gn.GridNetworkNoisy(M,lvalues[i])
        gridNets.append(network)

    x = np.linspace(0,maxPoint,numPoints)

    colors = ['tab:blue','tab:green','tab:red','tab:orange']

    for i in range(network.M):
        for j in range(len(gridNets)):
            responses = gridNets[j].r_error_free(x, i)
            plt.plot(x, responses,color=colors[j])

    plt.yticks([0,1])
    plt.ylim([0,1])
    plt.xlim([0,maxPoint])
    plt.tick_params(bottom = False,labelbottom=False)

    #Network 2
    plt.subplot(212)
    lvalues = lvalues2
    gridNets = []
    for i in range(N):
        network=gn.GridNetworkNoisy(M,lvalues[i])
        gridNets.append(network)

    for i in range(network.M):
        for j in range(len(gridNets)):
            responses = gridNets[j].r_error_free(x, i)
            plt.plot(x, responses,color=colors[j])

    plt.yticks([0,1])
    plt.ylim([0,1])
    plt.xlim([0,maxPoint])
    plt.tick_params(bottom = False,labelbottom=False)

    plt.show()

#Plots tuning curves for readout cells over distance
def testG(readout):
    x = np.linspace(0,readout.Rl,1000)
    y = np.zeros(1000)
    
    for j in range(readout.R):
        i = 0
        for location in x:
            y[i] = readout.readoutTermforWeights(location,j)
            i = i+1
        plt.plot(x,y)
    

    plt.plot(x,y)
    plt.show()


#Plots the error in coded distance for ideal response (no noise)
#over Rl for some time step
def plotErrorLambda(readout):
    preferredPhases = readout.readoutPreferred

    x = np.arange(.1,4,.1)
    err = np.zeros(len(x))
    k = 0
    for i in x:
        h0=np.zeros(readout.R)
        trueans = find_nearest(preferredPhases,i)
        for j in range(readout.R): 
            h0[j]=readout.summedInputstoReadout_error_free(j,i)
        
        ans = readout.Readout(h0)

        if ans != trueans:
            err[k] = np.abs(ans-trueans)
        k = k+1
        
    plt.plot(x,err,color='tab:blue')
    plt.show()
    

#Plots readout cells, ideal and error corrected
#For some input location x
def testReadout(readout,x):
    # start prediction
    h=np.zeros(readout.R) # prediction init
    h0=np.zeros(readout.R) # true readout init
    # with noisy and error correction
    plt.subplot(211)
    for i in range(readout.R): 
        h[i]=readout.summedInputstoReadout(i,x,40) # readout prediction from gridcell networks
    plt.plot(range(readout.R),h)
    # error free
    plt.subplot(212)
    for i in range(readout.R): 
        h0[i]=readout.summedInputstoReadout_error_free(i,x)
    plt.plot(range(readout.R),h0)
    plt.show()

    print("The winner readout cell is",readout.Readout(h))
    print("The true readout cell is",readout.Readout(h0))



M = 6
l = 1
network = gn.GridNetworkNoisy(M, l)

#gridCellPlot(network,4)

#plotR(4,8,[.3,.36,.45,.51],[.8,.6,.4,.2])

N = 4 #Number of networks
M = 8 #Number of neurons in a network
Rl = 4 #Distance max
R = 40 #Number of readout cells
lvalues = [.3,.36,.45,.51]
fullNetwork = cn.FullGridNetworkwithReadout(N,M,Rl,R,lvalues)

#testG(fullNetwork)
#plotErrorLambda(fullNetwork)
testReadout(fullNetwork,1)



