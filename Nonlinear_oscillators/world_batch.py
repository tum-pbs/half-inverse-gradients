import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


def l2_loss(a,b):
    diff = a-b
    loss_batch = tf.reduce_sum(diff**2,axis=1)
    loss = tf.reduce_sum(loss_batch)
    return loss,loss_batch

class World_batch():
    def __init__(self, ODE, time_int):
        self.ODE = ODE
        self.time_int = time_int

    def set_init_field(self, x0):
        self.x0 = x0


    def step(self, dt, control_t):
        self.x = self.time_int(self.x, dt, control_t, self.ODE)


    def time_evolution(self, dt, Nt, control):
        self.x_data = []
        self.x_data.append(self.x0)
        self.x = self.x0
        for i in range(Nt):
            control_t = control[:,i]
            self.step(dt, control_t)
            self.x_data.append(self.x)

    def plot(self):
        data = self.x_data
        for i in range(data.shape[2]):
            plt.plot(data[:, 0, i],label="x"+str(i))
            plt.plot(data[:, 1, i],label="v"+str(i))
        plt.plot(self.control,label="control")
        plt.legend()
        plt.show()

    def get_solver_dp(self,dt,Nt):
        @tf.function
        def solver(x0, control):
            self.set_init_field(x0)
            self.time_evolution(dt,Nt,control)
            return self.x

        return solver
