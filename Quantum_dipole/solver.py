import numpy as np
import tensorflow as tf
from tensorflow.python.ops import gen_array_ops
import matplotlib.pyplot as plt


def normalize_probability(psi):
    prob = np.abs(psi) ** 2
    total_prob = np.sum(prob)
    res = psi / np.sqrt(total_prob)
    return res

def normalize_probability_batch(psi_batch):
    prob_batch = np.abs(psi_batch) ** 2
    total_prob_batch = np.sum(prob_batch,axis=1)
    a = np.reshape(np.sqrt(total_prob_batch),(-1,1))
    b = np.reshape(np.ones(psi_batch.shape[1:]), (1, -1))
    div = np.kron(a, b)
    res = psi_batch / div
    return res

def projection_correction(ip_batch):
    return 1/tf.abs(tf.math.real(ip_batch))

@tf.function
def to_real(a_c):
    a_r = tf.stack([tf.math.real(a_c), tf.math.imag(a_c)], axis=-1)
    return a_r

@tf.function
def to_complex(a_r):
    k = tf.cast(a_r, tf.complex64)
    a_c = k[:, :, 0] + 1.j * k[:, :, 1]
    return a_c

@tf.function
def ip_loss(ar,br):
    a = to_complex(ar)
    b = to_complex(br)
    ip_batch = tf.reduce_sum(tf.math.conj(a)*b,axis=1)
    loss_dp = 1-tf.abs(ip_batch)**2
    loss = tf.reduce_sum(loss_dp)
    return loss,loss_dp, ip_batch


@tf.function
def l2_loss(x, y):
    l = 0.5 * tf.reduce_sum((x - y) ** 2)
    return l


@tf.function
def qm_step_batch(psi_batch_r, control_batch, dt, dx,xmin,xmax):

    psi_batch_c = to_complex(psi_batch_r)

    control=tf.reshape(control_batch,(-1,1))

    therange=tf.reshape(tf.range(xmin, xmax, delta=dx, dtype=tf.float32, name='range')[1:],(1,-1))
    pot_batch=0.5j * dt * tf.cast(tf.tensordot(control,therange,axes=(1,0)),tf.complex64)


    batch_size =  psi_batch_c.shape[0]
    spatial_size = psi_batch_c.shape[1]

    alpha_batch = 1.j*(0.5 * dt * tf.ones((batch_size,spatial_size), dtype=tf.complex64) / dx / dx)
    gamma_batch = tf.ones((batch_size,spatial_size), dtype=tf.complex64) - 1.j * dt / dx / dx
    eta_batch = tf.ones((batch_size,spatial_size), dtype=tf.complex64) + 1.j * dt / dx / dx

    U_2_diag = gamma_batch - pot_batch
    U_2_subdiag = alpha_batch
    U_2_stack = tf.stack([U_2_subdiag,U_2_diag,U_2_subdiag],axis=1)
    U_2_batch = gen_array_ops.matrix_diag_v2(U_2_stack, k=(-1, 1), num_rows=-1, num_cols=-1, padding_value=0)

    U_1_diag = eta_batch + pot_batch
    U_1_subdiag = - alpha_batch
    U_1_stack = tf.stack([U_1_subdiag,U_1_diag,U_1_subdiag],axis=1)
    U_1_batch = gen_array_ops.matrix_diag_v2(U_1_stack, k=(-1, 1), num_rows=-1, num_cols=-1, padding_value=0)

    psi_batch_1 = tf.expand_dims(psi_batch_c,-1)

    b_batch = tf.tensordot(U_2_batch, psi_batch_1,axes=(2,1))
    b_batch1 = tf.transpose(b_batch,perm=(1,3,0,2))
    b_batch2 = tf.linalg.diag_part(b_batch1)
    b_batch3 = tf.transpose(b_batch2,perm=(2,0,1))

    psi_t_batch = tf.linalg.solve(U_1_batch, b_batch3)[:,:,0]

    psi_t_batch_r= to_real(psi_t_batch)
    return psi_t_batch_r

def eigenstate(n,L):

    L = L + 2
    points = np.arange(L, dtype=np.complex64)
    k_n = np.pi * n / (L - 1)
    wave = np.sin(k_n * points)  
    wave = normalize_probability(wave)
    return wave[1:-1]

es=eigenstate

class World_batch():
    def __init__(self, qm_step,xmin,xmax,Nx):
        self.qm_step = qm_step
        self.xmin = xmin
        self.xmax = xmax
        self.Nx = Nx
        self.dx = (xmax - xmin) / (Nx - 1)

    def set_init_field(self, x0):
        self.x0 = x0


    def step(self, dt, control_t):
        self.x = self.qm_step(self.x, control_t, dt, self.dx, self.xmin, self.xmax)

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
