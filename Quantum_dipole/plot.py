import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import pickle
import os




simulation_names = ['QMC_Adam','QMC_HIG']


path = os.path.dirname(os.path.realpath(__file__))+'/'


for simulation_name in simulation_names:

    try:
        print(simulation_name)
        file = open(path+simulation_name+'_RESULTS.pickle', "rb")
        results = pickle.load(file)
        file.close()


        sim_time = results['simulation_time']
        loss = np.array(results['test_loss'])

        if loss.shape[0]>200:
            max_points = 100
            loss = loss[::loss.shape[0] // max_points]
            sim_time = sim_time[::sim_time.shape[0] // max_points]

        label = simulation_name


        plt.plot(sim_time,loss,label=label)

        

    except (FileNotFoundError,IndexError) as e:
        print(e)


plt.title('Quantum Dipole',fontsize=16)
plt.xlabel('Wall clock time [sec]',fontsize=16)
plt.ylabel('Loss',fontsize=16)
plt.yscale('log')
plt.legend(loc=1,fontsize=12)

plt.tight_layout()
plt.savefig(path+"plot.png")

