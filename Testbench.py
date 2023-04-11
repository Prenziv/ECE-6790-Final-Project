#Testbench
import BasicGridCell as g
import NoisyGridNetwork as gn
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import time

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
    neuron1 = g.GridNetworkNoiseless(5,4)
    neuron2 = g.GridNetworkNoiseless(5,4)

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
   t = np.arange(0,2,.2)
   
   network = gn.GridNetworkNoisy(5,5)
   network.accumulatedNoise = 0

   x = np.linspace(0,2*5,1000)
   y = network.r(x,3,0,.2)
   plt.ion()
   figure, ax = plt.subplots(figsize=(10, 8))
   line1, = ax.plot(x, y)
   line2, = ax.plot(x,y)

   for i in range(10):
       newY = network.r(x,3,t[i],.2)

       line1.set_xdata(x)
       line1.set_ydata(newY)

       figure.canvas.draw()

       figure.canvas.flush_events()

       time.sleep(1)


testNoise()