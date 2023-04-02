import numpy as np
import math
import matplotlib.pyplot as plt
import numpy.matlib


def gc_dynamics_1d(periodic, N):
    #----------------------------------------------
    #network parameters
    m = 4  #CV = 1/sqrt(m)
    x_prefs = np.arange(N).T / N  #inherited location preferences (m)

    #FF input
    beta_vel = 1.5  #velocity gain
    beta_0 = 70  #uniform input
    alpha = 1000  #weight imbalance parameter
    gamma = 1.05 / 100  #Cennter Surround weight params
    beta = 1 / 100  #Cennter Surround weight params

    #temporal parameters

    T = 100  #length of integration time blocks (s)
    dt = 1 / 2000  #step size of numerical integration (s)
    tau_s = 30 / 1000  #synaptic time constant (s)

    #Graphing parameters
    # bins = np.linspace(0 + .01, 50, 1 - .01)

    # Trajectory Data (Sinusoidal)
    x = (np.sin(np.arange(dt, T, dt) * 2 * np.pi / 10) + 1) / 2
    v = np.zeros(int(T / dt))
    for i in range(1, int(T / dt) - 1):
        v[i] = (x[i] - x[i - 1]) / dt

    z = range(int(-N / 2), int(N / 2), 1)
    z = np.array(z)
    # Feed forward network input
    if periodic == 1:
        # gaussian FF input for aperiodic network
        envelope = np.exp(-4 * z.T / np.power((N / 2), 2))
    else:
        envelope = np.ones((N, 1))

    s_prev = np.zeros(2 * N)  #Population activity
    spk = np.zeros((2 * N, int(T / dt)))  #Total spiking
    spk_count = np.zeros((2 * N, 1))  #Current spiking

    # Weight setup
    crossSection = alpha * (np.exp(-gamma * pow(z, 2)) -
                            np.exp(-beta * pow(z, 2)))
    crossSection = np.roll(crossSection, (0, int(N / 2) - 1))

    W_RR = np.zeros((N, N))
    W_LL = np.zeros((N, N))
    W_RL = np.zeros((N, N))
    W_LR = np.zeros((N, N))

    for i in range(N):
        W_RR[i, :] = np.roll(crossSection,
                             (0, i - 1))  # Right neurons to Right neurons
        W_LL[i, :] = np.roll(crossSection,
                             (0, i + 1))  # Left neurons to Left neurons
        W_RL[i, :] = np.roll(crossSection,
                             (0, i))  # Left neurons to Right neurons
        W_LR[i, :] = np.roll(crossSection,
                             (0, i))  # Right neurons to Left neurons

    for t in range(1, int(T / dt)):
        #LEFT population
        v_L = (1 - beta_vel * v[t])
        g_LL = W_LL * s_prev[0:N]  #L->L
        g_LR = W_LR * s_prev[N:2 * N]  #R->L
        G_L = v_L * ((g_LL + g_LR) + envelope * beta_0
                     )  #input conductance into Left population

        #RIGHT population
        v_R = (1 + beta_vel * v[t])
        g_RR = W_RR * s_prev[N:2 * N]  #R->R
        g_RL = W_RL * s_prev[0:N]  #L->R
        G_R = v_R * ((g_RR + g_RL) + envelope * beta_0
                     )  #input conductance into Right population

        G = np.concatenate((G_L, G_R))
        F = np.zeros((2 * N, 1)) + G * (G >= 0)  #linear transfer function

        # subdivide interval m times
        spk_sub = np.random.poisson(np.matlib.repmat(F, 1, m) * dt)
        spk_count = spk_count + np.sum(spk_sub, 1).reshape(spk_count.shape)
        spk[:, t] = np.floor(spk_count / m).reshape(spk_count.shape[0])
        spk_count = spk_count % m

        #update population activity
        s_new = s_prev + spk[:, t] - s_prev * dt / tau_s
        s_prev = s_new

        if (t % 100 == 0):  #plot every 100 steps
            plt.subplot(2, 2, 1)
            plt.plot(x_prefs, W_RR[:, int(N / 2)], 'r')
            plt.plot(x_prefs, W_LL[:, int(N / 2)], 'b')
            plt.title('Intra Connections')
            plt.subplot(2, 2, 2)
            plt.plot(x_prefs, F[0:N])
            plt.title('Population Response')
            plt.subplot(2, 2, 3)
            plt.plot(
                x_prefs,
                np.exp(-pow((x_prefs - x[t]), 2) / pow(0.001, 2)) /
                max(np.exp(-pow((x_prefs - x[t]), 2) / pow(.001, 2))), 'g')
            plt.title('Position')
            # plt.subplot(2,2,4)
            # plt.plot.hist(bins,histc(x(1:t).*spk(N/2,1:t),bins)/dt./histc(x(1:t),bins))
            # plt.title('SN Response')
            plt.show()


gc_dynamics_1d(1, 128)
