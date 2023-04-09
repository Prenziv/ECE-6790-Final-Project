#Testbench
import BasicGridCell as g
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt


def testPhi(neuron):
    l = neuron.l

    x = np.linspace(0,2*l,100)
    phaseResponse = neuron.phi(x)

    plt.plot(x[0:50],phaseResponse[0:50])
    plt.plot(x[50:99],phaseResponse[50:99],color = 'tab:blue')
    plt.text(0,.7,'$\lambda = $'+str(l))
    plt.xlabel('Location')
    plt.ylabel('Spatial Phase')
    plt.show()


def testR(neuron1,neuron2):
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


#Network with M = 5, lambda = 4
network1 = g.GridNetworkNoiseless(5,4)
#Network with M = 4, lambda = 8
network2 = g.GridNetworkNoiseless(5,8)

testR(network1,network2)