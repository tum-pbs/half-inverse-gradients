import numpy as np
import tensorflow as tf


def build_laplace(n,boundary='0'):
    if n==1:
        return np.zeros((1,1),dtype=np.float32)
    d1 = -2 * np.ones((n,),dtype=np.float32)
    d2 = 1 * np.ones((n-1,),dtype=np.float32)
    lap = np.zeros((n,n),dtype=np.float32)
    lap[range(n),range(n)]=d1
    lap[range(1,n),range(n-1)]=d2
    lap[range(n-1),range(1,n)]=d2
    if boundary=='0':
        lap[0,0]=lap[n-1,n-1]=-1

    return lap


def lower_build_laplace(n,boundary='0'):
    if n==1:
        return np.zeros((1,1),dtype=np.float32)
    d1 = -1 * np.ones((n,),dtype=np.float32)
    d2 = np.ones((n-1,),dtype=np.float32)
    lap = np.zeros((n,n),dtype=np.float32)
    lap[range(n),range(n)]=d1
    lap[range(1,n),range(n-1)]=d2
    if boundary=='0':
        lap[0,0]=0
    return lap

def upper_build_laplace(n,boundary='0'):
    if n==1:
        return np.zeros((1,1),dtype=np.float32)
    d1 = -1 * np.ones((n,),dtype=np.float32)
    d2 = np.ones((n-1,),dtype=np.float32)
    lap = np.zeros((n,n),dtype=np.float32)
    lap[range(n),range(n)]=d1
    lap[range(n-1),range(1,n)]=d2
    if boundary=='0':
        lap[n-1,n-1]=0
    return lap


def eigenstate(n,i):
    lap = build_laplace(n)
    ew, ev = np.linalg.eigh(lap)
    res = ev[:,i]
    return res

def coeffs(n):
    a = np.zeros((n,))
    a[-1]=1
    coefs = []
    for i in range(n):
        coefs.append(eigenstate(n,i)@a)
    print(coefs)




@tf.function
def coup_nonlin_osc_batch( x, control):
    n_osc = x.shape[1]//2

    # natural time evo
    a1 = np.array([[0,1],[-1,0]],dtype=np.float32)
    a2 = np.eye(n_osc,dtype=np.float32)
    A = np.kron(a1,a2)
    x_dot1 = tf.tensordot(x,A,axes = (1,1))

    # linear interaction term
    interaction_strength = 0.0
    b1 = np.array([[0,0],[1,0]],dtype=np.float32)
    b2 = build_laplace(n_osc)
    B = interaction_strength * np.kron(b1,b2)
    x_dot2 = tf.tensordot(x,B, axes=(1, 1))

    # control term
    control_vector = np.zeros((n_osc,),dtype=np.float32)
    control_vector[-1] = 3.0
    c1 = np.array([0,1],dtype=np.float32)
    c2 = control_vector
    C = np.kron(c1,c2)
    x_dot3 = tf.tensordot(control,C, axes=0)

    # cubic interaction term
    cubic_interaction_strength = 1.0
    d1 = np.array([[0,0],[1,0]],dtype=np.float32)
    d2a = upper_build_laplace(n_osc)
    d2b = lower_build_laplace(n_osc)
    Da = cubic_interaction_strength * np.kron(d1,d2a)
    Db = cubic_interaction_strength * np.kron(d1, d2b)
    x_diffa = tf.tensordot(x,Da, axes=(1, 1))
    x_diffb = tf.tensordot(x,Db, axes=(1, 1))
    x_dot4 = cubic_interaction_strength * (x_diffa ** 3 + x_diffb ** 3)
    
    #all terms
    x_dot = x_dot1 + x_dot2 +x_dot3 +x_dot4
    return x_dot