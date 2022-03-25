import numpy as np
import tensorflow as tf
from simulation import simulation
import pickle
import os



simulation_name = 'NLO_HIG'


batch_size = 128
mode = 'IG'
learning_rate = 1.0
truncation = 10**-6
inversion = 'HIG'
opt = tf.keras.optimizers.SGD
seed = 22

Nx = 2                  # Number of oscillators
Nt = 96                 # Number of time steps
dt = 0.125              # Time steps
N = 4096                # Number of data points
criteria = 'sim_time'   # Stopping criteria
maxnumber = 60*60       # Simulation time in seconds


sim = simulation()
sim.set_physics(Nx,Nt,dt)
sim.set_optimization(mode,opt,truncation,inversion)
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

