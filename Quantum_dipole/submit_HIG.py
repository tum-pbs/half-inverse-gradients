import numpy as np
import tensorflow as tf
from simulation import simulation
import pickle
import os



simulation_name = 'QMC_HIG'

seed = 1
batch_size = 16
mode = 'HIG'
learning_rate = 0.5
truncation = 10**-5

opt = tf.keras.optimizers.SGD

Nx = 16                 # Spatial discretization
Nt = 96*4               # Number of time steps
dt = 0.05               # Time step
N = 1024                # Number of data points
maxnumber = 60*90       # Simulation time in seconds
criteria = 'sim_time'   # Stopping criteria




           


sim = simulation()
sim.set_physics(Nx,Nt,dt)
sim.set_optimization(mode,opt,truncation)
sim.set_learning(N,batch_size,learning_rate,criteria,maxnumber)
sim.set_seed(seed)

results, sim_par = sim.start()




path = os.path.dirname(os.path.realpath(__file__))+'/'




file1 = open(path+simulation_name+'_RESULTS.pickle', "wb")
pickle.dump(results, file1)
file1.close()


with open(path+simulation_name+'_PARAMETERS.txt', "w") as text_file:
    for key in sim_par.keys():
        print(key+':',sim_par[key] ,file=text_file)

print('FINISHED')


