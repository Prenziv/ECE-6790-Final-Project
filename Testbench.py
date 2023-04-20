#Testbench
import BasicGridCell as g
import NoisyGridNetwork as gn
import numpy as np
import matplotlib.pyplot as plt
import time
import ReadoutCell as ro

#Shows phi response of grid cell network over two periods
def testPhi(m,l):
    neuron = g.GridNetworkNoiseless(m,l)
    l = neuron.l

    x = np.linspace(0,2*l,100)
    phaseResponse = neuron.phi(x)

    plt.plot(x[0:50],phaseResponse[0:50])
    plt.plot(x[50:99],phaseResponse[50:99],color = 'tab:blue')
    plt.text(0,.7,'$\lambda = $'+str(l))
    plt.xlabel('Location')
    plt.ylabel('Spatial Phase')
    plt.show()

#Shows neural tuning curve responses of two networks of neurons
def testR():
    neuron1 = g.GridNetworkNoiseless(8,4)
    neuron2 = g.GridNetworkNoiseless(4,4)

    l1 = neuron1.l
    m1 = neuron1.M

    l2 = neuron2.l
    m2 = neuron2.M

    x = np.linspace(0,2*l2,1000)

    plt.figure()
    plt.subplot(211)
    for i in range(m1):
        responses = neuron1.r(x,i)
        plt.plot(x,responses)

    plt.subplot(212)
    for i in range(m2):
        responses = neuron2.r(x,i)
        plt.plot(x,responses)

    plt.show()

#Shows noisy response of neuron
def testNoise():
    
    network = gn.GridNetworkNoisy(5,5)
    network.generateNoise(25)

    x = np.linspace(0,2*5,1000)
    y = [] # neural tuning curve of one neuron in the network for some distance x
    for i in range(network.M):
            y.append(network.r(x,i,0))

    plt.ion()
    figure, ax = plt.subplots(figsize=(10, 8))

    lines = []
    for i in range(network.M):
            lines.append(ax.plot(x,y[i],'r'))
    
    lines2 = []
    for i in range(network.M):
            lines2.append(ax.plot(x,y[i],'b'))

    for i in range(25):
        for j in range(network.M):
            lines[j][0].set_xdata(x)
            lines[j][0].set_ydata(network.r(x,j,i))
        

        figure.canvas.draw()
        figure.canvas.flush_events()

        time.sleep(2)


def testReadout():
    M=3 # networkNum=2
    N=5 # neuron num
    test_t=50 # The instantaneous time we predict
    # grid cell networks init
    gridNets=[]
    for i in range(M):
        network=gn.GridNetworkNoisy(N,N)
        network.generateNoise(100)
        gridNets.append(network)
    # readout cell init
    readout= ro.ReadoutCell(M,N)
    # start prediction
    h=np.zeros(readout.R) # prediction init
    h0=np.zeros(readout.R) # true readout init
    # with noisy and error correction
    plt.subplot(211)
    for i in range(readout.R): 
        h[i]=readout.summedInputstoReadout(i,gridNets,test_t) # readout prediction from gridcell networks
    plt.plot(range(readout.R),h)
    # error free
    plt.subplot(212)
    for i in range(readout.R): 
        h0[i]=readout.summedInputstoReadout_error_free(i,gridNets,test_t)
    plt.plot(range(readout.R),h0)
    plt.show()
    print("The winner readout cell is",readout.Readout(h))
    print("The true readout cell is",readout.Readout(h0))
    
    
    

# testNoise()
# testR()
testReadout()