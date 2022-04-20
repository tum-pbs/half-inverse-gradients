import numpy as np
import matplotlib.pyplot as plt
import os

path = os.path.dirname(os.path.realpath(__file__))+'/'


sim_path = '/home/user/phi/submit_Poisson/'
sims = ['sim_000000','sim_000001']
labels = ['Adam','HIG']


for i,sim in enumerate(sims):
    try:
        step_time = np.loadtxt(sim_path+sim+'/log_step_time.txt')[:,1]
        y_loss_l2 = np.loadtxt(sim_path+sim+'/log_loss_'+labels[i]+'_0.txt')[:,1]

        wall_clock_time  = np.cumsum(step_time)
        plt.plot(wall_clock_time,y_loss_l2,label=labels[i])
        
    except (FileNotFoundError,IndexError) as e:
        print(e)

plt.xscale('log')
plt.yscale('log')
plt.title('Nonlinear Oscillators',fontsize=16)
plt.xlabel('Wall clock time [sec]',fontsize=16)
plt.ylabel('Loss',fontsize=16)
plt.legend(loc=1,fontsize=12)
plt.tight_layout()
plt.savefig(path+"plot.png")